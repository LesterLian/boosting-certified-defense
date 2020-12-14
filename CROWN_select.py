import argparse
import pickle
import sys
import os

import torch

from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, WeightedDataLoader
from argparser import argparser
from config import load_config, config_modelloader, config_dataloader


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T, shuffle=False)
        # self.weighted_data = torch.utils.data.DataLoader(self.weighted_data, batch_size=256, shuffle=False)


# class CROWNPredictor(BasePredictor):
#     def __init__(self, model):
#         super(CROWNPredictor, self).__init__(model)
#
#     def predict(self, X):
#         X = X.to('cuda')
#         output = self.model(X)
#         return output


def main():
    config = load_config(crown_args)
    global_eval_config = config["eval_params"]

    if args.models is not None:
        config["models_path"] = args.models
    models, model_names = config_modelloader(config, load_pretrain=True)
    models = [BasePredictor(model) for model in models]
    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_eval_config["loader_params"])
    # Boost
    if args.load_ada is None:
        ada = PretrainedSAMME(train_data,
                              models,
                              T=args.iteration)

        ada.train()
        # Save the ensemble model
        dump_name = os.path.basename(os.path.splitext(crown_args.config)[0])
        ada.weighted_data = None
        ada.base_predictor_list = None
        with open(f'../ada_{dump_name}_T{ada.T}', "wb") as f:
            pickle.dump(ada, f)
    else:
        ada = pickle.load(open(args.load_ada, 'rb'))
        ada.base_predictor_list = [BasePredictor(model) for model in models]
    print(ada.predictor_list)
    print(ada.predictor_weight)

    print(f'\n{"Testing":=^20}\n')
    with torch.no_grad():
        ada.base_predictor_list = models
        base_max = 0
        for idx in ada.predictor_list:
            predictor = ada.base_predictor_list[idx]
            correct = 0
            for X, y in test_data:
                X = X.to(ada.device)
                predictor.model.to(ada.device)
                y_pred = predictor.predict(X).to('cpu')
                correct += (y_pred == y).sum()
            if correct > base_max:
                base_max = correct
        correct = 0
        for X, y in test_data:
            y_pred = ada.predict(X).to('cpu')
            correct += (y_pred == y).sum()
        print(f'max clean accuracy of base model: {base_max/len(test_data.dataset)}')
        print(f'clean accuracy: {correct/len(test_data.dataset)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--iteration', '-T', type=int,
                        help='maximum number of running Adaboost')
    parser.add_argument('--load_ada', '-l', type=str,
                        help='path containing the saved Adaboost model')
    parser.add_argument('--models', '-m', type=str,
                        help='path containing the pretrained base model')

    args, unknown = parser.parse_known_args()
    # Remove parsed args and pass to CORWN-IBP
    sys.argv[1:] = unknown
    crown_args = argparser()

    main()
