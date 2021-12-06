import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ada import AdaBoostBase, AdaBoostTrain, AdaBoostSamme, BasePredictor, WeightedDataLoader, get_weighted_ce
from argparser import argparser
from bound_layers import BoundSequential, BoundDataParallel
from config import load_config, config_modelloader, config_dataloader, get_path
from train import Train, Logger
from train_ada import Train as TrainAda
from eps_scheduler import EpsilonScheduler


class CROWNPredictor(BasePredictor):
    def predict(self, X):
        return self.model(X, method_opt="forward", disable_multi_gpu=True).argmax(dim=1)


class TrainSAMME(AdaBoostSamme):
    def __init__(self, train_data, test_data, base_predictor_list, T, config):
        super(TrainSAMME, self).__init__(train_data, base_predictor_list, T, shuffle=True)
        # self.train_data = train_data
        self.train_data = self.weighted_data
        self.test_data = test_data
        self.config = config
        # train.CrossEntropyLoss = get_weighted_ce(self.distribution)

    def predict(self, X):
        final_pred = torch.zeros((len(X), self.K)).cuda()
        X = X.to(self.device)
        for i, weight in zip(self.predictor_list, self.predictor_weight):
            cur_predictor = self.base_predictor_list[i]
            cur_predictor.model.eval()
            cur_weight = weight.cuda()
            final_pred += cur_weight * cur_predictor.model(X, method_opt="forward", disable_multi_gpu=True)

        return final_pred.argmax(dim=1)

    def gen_new_base_predictor(self, cur_round):
        # for model, model_config in zip(models, config["models"]):
        model = self.base_predictor_list[cur_round]
        # make a copy of global training config, and update per-model config
        # train_config = copy.deepcopy(global_train_config)
        train_config = self.config["training_params"]
        config = self.config
        model_id = str(cur_round)
        # if "training_params" in model_config:
        #     train_config = update_dict(train_config, model_config["training_params"])

        # read training parameters from config file
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_milestones = train_config["lr_decay_milestones"]
        lr_decay_factor = train_config["lr_decay_factor"]
        multi_gpu = train_config["multi_gpu"]
        # parameters specific to a training method
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])

        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")

        batch_multiplier = train_config["method_params"].get("batch_multiplier", 1)
        batch_size = self.train_data.batch_size * batch_multiplier
        num_steps_per_epoch = int(np.ceil(1.0 * len(self.train_data.dataset) / batch_size))
        epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"),
                                             schedule_start * num_steps_per_epoch,
                                             ((schedule_start + schedule_length) - 1) * num_steps_per_epoch,
                                             starting_epsilon, end_epsilon, num_steps_per_epoch)
        max_eps = end_epsilon

        if lr_decay_step:
            # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
            lr_decay_milestones = None
        elif lr_decay_milestones:
            # Decay learning rate by lr_decay_factor at a few milestones.
            lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones,
                                                          gamma=lr_decay_factor)
        else:
            raise ValueError("one of lr_decay_step and lr_decay_milestones must be not empty.")
        model_name = get_path(config, model_id, "model", load=False)
        best_model_name = get_path(config, model_id, "best_model", load=False)
        model_log = get_path(config, model_id, "train_log")
        logger = Logger(open(model_log, "w"))
        logger.log(model_name)
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", self.train_data.std)
        best_err = np.inf
        recorded_clean_err = np.inf
        timer = 0.0

        if multi_gpu:
            logger.log("\nUsing multiple GPUs for computing CROWN-IBP bounds\n")
            model = BoundDataParallel(model)
        model = model.cuda()

        for t in range(epochs):  # TODO epochs
            epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
            epoch_end_eps = epsilon_scheduler.get_eps(t + 1, 0)
            logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(),
                                                                                    epoch_start_eps, epoch_end_eps))
            # with torch.autograd.detect_anomaly():
            start_time = time.time()
            TrainAda(model, t, self.train_data, epsilon_scheduler, max_eps,
                   norm, logger, verbose, True, opt,
                        method,
                        **method_param)
            if lr_decay_step:
                # Use stepLR. Note that we manually set up epoch number here, so the +1 offset.
                lr_scheduler.step(epoch=max(t - (schedule_start + schedule_length - 1) + 1, 0))
            elif lr_decay_milestones:
                # Use MultiStepLR with milestones.
                lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            with torch.no_grad():
                # evaluate
                err, clean_err, _ = Train(model, t, self.test_data,
                                             EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1),
                                             max_eps,
                                             norm, logger, verbose, False, None, method, **method_param)

            logger.log('saving to', model_name)
            torch.save({
                'state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
                'epoch': t,
            }, model_name)

            # TODO: what's this
            # save the best model after we reached the schedule
            if t >= (schedule_start + schedule_length):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    logger.log('Saving best model {} with error {}'.format(best_model_name, best_err))
                    torch.save({
                        'state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
                        'robust_err': err,
                        'clean_err': clean_err,
                        'epoch': t,
                    }, best_model_name)
        # Same code from select
        predictor = CROWNPredictor(model)
        self.base_predictor_list[cur_round] = predictor
        incorrect_pred = None
        index = None
        error = 0
        for X, y, w, idx in tqdm(self.weighted_data, total=len(self.weighted_data)):
            X = X.to(self.device)
            y_pred = predictor.predict(X).to('cpu')
            error += ((y_pred != y) * w).sum()
            if incorrect_pred is None:
                incorrect_pred = y_pred != y
                index = idx
            else:
                incorrect_pred = torch.cat((incorrect_pred, y_pred != y), dim=0)
                index = torch.cat((index, idx), dim=0)
        return cur_round, error, incorrect_pred, index


# class CROWNPredictor(BasePredictor):
#     def __init__(self, model):
#         super(CROWNPredictor, self).__init__(model)
#
#     def train(self, weighted_dataset):
#


def main():
    config = load_config(crown_args)
    global_train_config = config["training_params"]
    if args.epsilon is not None:
        global_train_config['epsilon'] = args.epsilon
    models, _ = config_modelloader(config)
    models = [BoundSequential.convert(model, global_train_config["method_params"]["bound_opts"])
              for model in models]

    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_train_config["loader_params"])
    # Read existing Adaboost ensemble
    dump_name = os.path.basename(os.path.splitext(crown_args.config)[0])
    dump_path = f'../ada_train_{dump_name}_eps{args.epsilon}_T{args.iteration}'
    # if os.path.isfile(dump_path):
    if os.path.isfile(dump_path):
        print(f'Found Adaboost model at {dump_path}')
        ada = pickle.load(open(dump_path, 'rb'))
    else:
        # Boost
        ada = TrainSAMME(train_data,
                         test_data,
                         # [BasePredictor(model) for model in models],
                         models,
                         T=args.iteration,
                         config=config)
        ada.train()

        print(f'Predictor Weights: {ada.predictor_weight}')

        # Save the ensemble model
        data = ada.weighted_data
        ada.weighted_data = None
        # ada.base_predictor_list = None
        with open(dump_path, "wb") as f:
            pickle.dump(ada, f)
        ada.weighted_data = data
    uniform = TrainSAMME(train_data,
                         test_data,
                         models,
                         T=args.iteration,
                         config=config)
    uniform.base_predictor_list = ada.base_predictor_list
    uniform.predictor_list = ada.predictor_list
    uniform.predictor_weight = torch.ones(args.iteration)

    # Evaluate Adaboost ensemble
    print(f'\n{"Testing":=^20}\n')
    with torch.no_grad():
        # ada.base_predictor_list = models
        base_max = 0
        base_min = 10000000
        for i in ada.predictor_list:
            predictor = ada.base_predictor_list[i]
            correct = 0
            for X, y in test_data:
                X = X.to(ada.device)
                predictor.model.to(ada.device)
                y_pred = predictor.predict(X).to('cpu')
                correct += (y_pred == y).sum()
            if correct > base_max:
                base_max = correct
                base_max_i = i
            if correct < base_min:
                base_min = correct
        correct = 0.0
        uniform_correct = 0.0
        for X, y in test_data:
            y_pred = ada.predict(X).to('cpu')
            correct += (y_pred == y).float().sum()

            y_pred = uniform.predict(X).to('cpu')
            uniform_correct += (y_pred == y).sum()
        print(f'max clean error of base model: '
              f'{1.0 - base_max / len(test_data.dataset)}')
        print(f'min clean error of base model: '
              f'{1.0 - base_min / len(test_data.dataset)}')
        print(f'uniform clean error: '
              f'{1.0 - uniform_correct / len(test_data.dataset)}')
        print(f'ada clean error: {1.0 - correct / len(test_data.dataset)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost training CROWN-IBP '
                                                 'models.')
    parser.add_argument('--iteration', '-T', type=int,
                        help='maximum number of running Adaboost')
    parser.add_argument('--epsilon', '-e', type=float,
                        help='epsilon for evaluating CROWN models')
    args, unknown = parser.parse_known_args()
    # Remove parsed args and pass to CORWN-IBP
    sys.argv[1:] = unknown
    crown_args = argparser()

    main()
