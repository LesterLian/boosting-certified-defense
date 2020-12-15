import argparse
import sys
import time

import numpy as np
import torch
from torch import optim

from ada import AdaBoostTrain, AdaBoostSamme, BasePredictor, WeightedDataLoader
from argparser import argparser
from bound_layers import BoundSequential, BoundDataParallel
from config import load_config, config_modelloader, config_dataloader, get_path
import train
from eps_scheduler import EpsilonScheduler


class TrainSAMME(AdaBoostTrain, AdaBoostSamme):
    def __init__(self, train_data, test_data, base_predictor_list, T, config):
        super(TrainSAMME, self).__init__(train_data, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10
        self.weighted_data = torch.utils.data.DataLoader(WeightedDataLoader(train_data, self.distribution))
        self.test_data = test_data
        self.config = config
        train.CrossEntropyLoss

    def gen_new_base_predictor(self, cur_round, weighted_train_dataset):
        # for model, model_config in zip(models, config["models"]):
        for model in self.base_predictor_list:
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
            batch_size = self.weighted_data.batch_size * batch_multiplier
            num_steps_per_epoch = int(np.ceil(1.0 * len(self.weighted_data.dataset) / batch_size))
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
            logger = train.Logger(open(model_log, "w"))
            logger.log(model_name)
            logger.log("Command line:", " ".join(sys.argv[:]))
            logger.log("training configurations:", train_config)
            logger.log("Model structure:")
            logger.log(str(model))
            logger.log("data std:", self.weighted_data.std)
            best_err = np.inf
            recorded_clean_err = np.inf
            timer = 0.0

            if multi_gpu:
                logger.log("\nUsing multiple GPUs for computing CROWN-IBP bounds\n")
                model = BoundDataParallel(model)
            model = model.cuda()

            for t in range(epochs):
                epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
                epoch_end_eps = epsilon_scheduler.get_eps(t + 1, 0)
                logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(),
                                                                                        epoch_start_eps, epoch_end_eps))
                # with torch.autograd.detect_anomaly():
                start_time = time.time()
                train.Train(model, t, self.weighted_data, epsilon_scheduler, max_eps, norm, logger, verbose, True, opt,
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
                    err, clean_err = train.Train(model, t, self.test_data,
                                                 EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1),
                                                 max_eps,
                                                 norm, logger, verbose, False, None, method, **method_param)

                logger.log('saving to', model_name)
                torch.save({
                    'state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
                    'epoch': t,
                }, model_name)

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


# class CROWNPredictor(BasePredictor):
#     def __init__(self, model):
#         super(CROWNPredictor, self).__init__(model)
#
#     def train(self, weighted_dataset):
#


def main():
    config = load_config(crown_args)
    global_train_config = config["training_params"]
    models, _ = config_modelloader(config)
    models = [BoundSequential.convert(model, global_train_config["method_params"]["bound_opts"])
              for model in models]

    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_train_config["loader_params"])
    # Boost
    ada = TrainSAMME(train_data.dataset,
                     test_data,
                     # [BasePredictor(model) for model in models],
                     models,
                     T=args.iteration,
                     config=config)
    ada.train()

    print(ada.predictor_list)
    print(ada.predictor_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--iteration', '-T', type=int,
                        help='the maximum number of running Adaboost')

    args, unknown = parser.parse_known_args()
    # Remove parsed args and pass to CORWN-IBP
    sys.argv[1:] = unknown
    crown_args = argparser()

    main()
