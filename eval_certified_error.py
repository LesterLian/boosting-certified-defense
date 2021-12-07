import argparse
import pickle
import sys
import os
import copy

import torch

from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, \
    WeightedDataLoader
from ada.modules import init_logger, log
from bound_layers import BoundSequential
from eps_scheduler import EpsilonScheduler
from argparser import argparser
from config import load_config, config_modelloader, config_dataloader
from train import Train, Logger
from train_ada import Train as TrainAda

from CROWN_train import CROWNPredictor, TrainSAMME
from CRWON_select_certified_error import PretrainedSAMME


def main():
    config = load_config(crown_args)
    global_eval_config = config["eval_params"]
    if args.epsilon is not None:
        global_eval_config['epsilon'] = args.epsilon

    # init_logger({
    #     "epsilon": global_eval_config['epsilon'],
    #     "T": args.iteration,
    # })
    csv = open(f'results.csv', 'a')
    result = f'{global_eval_config["epsilon"]},{args.iteration},'

    if args.models is not None:
        config["models_path"] = args.models
    config['models'] = [{**model, 'model_id': str(i)} for i, model in
                        enumerate(config['models'])]
    models, model_names = config_modelloader(config, load_pretrain=True)
    models = [BasePredictor(model) for model in models]
    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_eval_config[
        "loader_params"])
    # Boost
    if args.load_ada is None:
        raise ValueError('Must pass in load_ada or -l')
        return
    ada = pickle.load(open(args.load_ada, 'rb'))
    if ada.base_predictor_list is None:
        ada.base_predictor_list = models
    ada.predictor_list = ada.predictor_list[:args.iteration]
    ada.predictor_weight = ada.predictor_weight[:args.iteration]
    model_log_name = f'{os.path.basename(args.load_ada)}'
    print(ada.predictor_list)
    print(ada.predictor_weight)
    uniform = PretrainedSAMME(train_data,
                              models,
                              T=args.iteration)
    uniform.base_predictor_list = ada.base_predictor_list
    uniform.predictor_list = ada.predictor_list
    uniform.predictor_weight = torch.ones(args.iteration)


    print(f'\n{"Eval clean error":=^20}\n')
    with torch.no_grad():
        base_max = 0
        base_min = 10000000
        for idx in ada.predictor_list:
            predictor = ada.base_predictor_list[idx]
            correct = 0.0
            for X, y in test_data:
                X = X.to(ada.device)
                predictor.model.to(ada.device)
                y_pred = predictor.predict(X).to('cpu')
                correct += (y_pred == y).sum()
            if correct > base_max:
                base_max = correct
            if correct < base_min:
                base_min = correct
        correct = 0.0
        uniform_correct = 0.0

        for X, y in test_data:
            y_pred = ada.predict(X).to('cpu')
            correct += (y_pred == y).float().sum()

            y_pred = uniform.predict(X).to('cpu')
            uniform_correct += (y_pred == y).float().sum()
            # X = X.to(ada.device)
            # y_pred = None
            # num_base_predictor = float(len(ada.base_predictor_list))
            # for predictor in ada.base_predictor_list:
            #     md = predictor.model.to(ada.device)
            #     if y_pred is None:
            #         y_pred = md(X) / num_base_predictor
            #     else:
            #         y_pred += md(X) / num_base_predictor
            # uniform_correct += (y_pred.cpu().argmax(dim=1) == y).float().sum()

        print(f'ada name = {args.load_ada}')
        print(
            f'max clean error of base model: {1.0 - base_max / len(test_data.dataset)}')
        print(
            f'min clean error of base model: {1.0 - base_min / len(test_data.dataset)}')
        print(
            f'uniform clean error: {1.0 - uniform_correct / len(test_data.dataset)}')
        print(f'ada clean error: {1.0 - correct / len(test_data.dataset)}')
        # log({'AdaBoost Clean': 1.0 - correct / len(test_data.dataset),
        #      'Uniform Clean': 1.0 - uniform_correct / len(test_data.dataset),
        #      'Single Best Clean': 1.0 - base_max / len(test_data.dataset),
        #      'Single Worst Clean': 1.0 - base_min / len(test_data.dataset)
        #      })
        result += f'{1.0 - correct / len(test_data.dataset)},' \
                  f'{1.0 - uniform_correct / len(test_data.dataset)},' \
                  f'{1.0 - base_max / len(test_data.dataset)},' \
                  f'{1.0 - base_min / len(test_data.dataset)},'

    # Evaluating certified accuracy
    print(f'\n{"Eval certified error":=^20}\n')
    # read training parameters from config file
    eval_config = copy.deepcopy(global_eval_config)
    method = eval_config["method"]
    verbose = eval_config["verbose"]
    eps = eval_config["epsilon"]
    # parameters specific to a training method
    method_param = eval_config["method_params"]
    norm = float(eval_config["norm"])

    weighted_sum_lb = None
    uniform_sum_lb = None
    uniform_weight = float(len(ada.base_predictor_list))
    max_robust_err = 0.0
    min_robust_err = 1.0

    with torch.no_grad():
        robust_errs = []
        errs = []

        for idx, base_model_idx in enumerate(ada.predictor_list):
            if args.iteration > 0 and idx >= args.iteration: break
            model = models[base_model_idx].model
            # make a copy of global training config, and update per-model config

            model = BoundSequential.convert(model,
                                            eval_config["method_params"][
                                                "bound_opts"])
            model = model.cuda()
            train_data, test_data = config_dataloader(config, **eval_config[
                "loader_params"])

            logger = Logger(open(f'./model_log/{model_log_name}_{idx}.log',
                                 "w"))
            # evaluate
            robust_err, err, model_lb_batch = Train(model, 0, test_data,
                                                    EpsilonScheduler("linear",
                                                                     0, 0, eps,
                                                                     eps, 1),
                                                    eps, norm, logger, verbose,
                                                    False, None, method,
                                                    **method_param)
            model_lb = torch.cat(model_lb_batch, 0)

            if weighted_sum_lb is None:
                weighted_sum_lb = ada.predictor_weight[idx] * model_lb
            else:
                weighted_sum_lb += ada.predictor_weight[idx] * model_lb
            if uniform_sum_lb is None:
                uniform_sum_lb = uniform_weight * model_lb
            else:
                uniform_sum_lb += uniform_weight * model_lb
                # print(f'lb shape = {model_lb.size()}')
            # print(f'model_id = {idx}, robust_err = {robust_err}, clean_err = {err}')
            if robust_err < min_robust_err:
                min_robust_err = robust_err
            if robust_err > max_robust_err:
                max_robust_err = robust_err
            robust_errs.append(robust_err)
            errs.append(err)

        ensemble_robust_error = torch.sum((weighted_sum_lb < 0).any(
            dim=1)).cpu().detach().numpy() / weighted_sum_lb.size(0)

        # Iterate through all base model is wrong
        # for idx, predictor in enumerate(ada.base_predictor_list):
        #     model = predictor.model
        #
        #     model = BoundSequential.convert(model,
        #                                     eval_config["method_params"][
        #                                         "bound_opts"])
        #     model = model.cuda()
        #     train_data, test_data = config_dataloader(config, **eval_config[
        #         "loader_params"])
        #
        #     logger = Logger(open(f'./model_log/{model_log_name}_base{idx}.log',
        #                          "w"))
        #     # evaluate
        #     robust_err, err, model_lb_batch = Train(model, 0, test_data,
        #                                             EpsilonScheduler("linear",
        #                                                              0, 0, eps,
        #                                                              eps, 1),
        #                                             eps, norm, logger, verbose,
        #                                             False, None, method,
        #                                             **method_param)
        #     model_lb = torch.cat(model_lb_batch, 0)
        #
        #     if uniform_sum_lb is None:
        #         uniform_sum_lb = uniform_weight * model_lb
        #     else:
        #         uniform_sum_lb += uniform_weight * model_lb
        #     robust_errs.append(robust_err)
        #     errs.append(err)

        uniform_robust_err = torch.sum((uniform_sum_lb < 0).any(
            dim=1)).cpu().detach().numpy() / uniform_sum_lb.size(0)

        print(f'ada name = {args.load_ada}')
        print(f'ada robust error = {ensemble_robust_error}')
        print(f'uniform robust error = {uniform_robust_err}')
        print(f'sigle best robust error = {min_robust_err}')
        print(f'single worst robust error = {max_robust_err}')
        # log({'AdaBoost Certified': ensemble_robust_error,
        #      'Uniform Certified': uniform_robust_err,
        #      'Single Best Certified': min_robust_err,
        #      'Single Worst Certified': max_robust_err
        #      })
        result += f'{ensemble_robust_error},' \
                  f'{uniform_robust_err},' \
                  f'{min_robust_err},' \
                  f'{max_robust_err}\n'
        csv.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--iteration', '-T', type=int,
                        help='maximum number of running Adaboost')
    parser.add_argument('--epsilon', '-e', type=float,
                        help='epsilon for evaluating CROWN models')
    parser.add_argument('--load_ada', '-l', type=str,
                        help='path containing the saved Adaboost model')
    parser.add_argument('--models', '-m', type=str,
                        help='path containing the pretrained base model')

    args, unknown = parser.parse_known_args()
    # Remove parsed args and pass to CORWN-IBP
    sys.argv[1:] = unknown
    crown_args = argparser()

    main()
