import argparse
import sys

import torch

from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, WeightedDataset
from argparser import argparser
from config import load_config, config_modelloader, config_dataloader


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10
#         self.weighted_data = CROWNDataset(self.dataset, self.distribution)
#
#
# class CROWNDataset(WeightedDataset):
#     def __init__(self, dataset, weight):
#         super(CROWNDataset, self).__init__(dataset, weight)
#
#     def __getitem__(self, item):
#         X, y = self.dataset[item]
#         elem = (X.unsqueeze(0), y, self.weight[item])
#
#         return elem


class CROWNPredictor(BasePredictor):
    def __init__(self, model):
        super(CROWNPredictor, self).__init__(model)

    def predict(self, X):
        output = self.model(X.unsqueeze(0))
        return torch.argmax(output)


def main():
    config = load_config(crown_args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain=True)

    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_eval_config["loader_params"])
    # Boost
    ada = PretrainedSAMME(train_data.dataset,
                          [CROWNPredictor(model) for model in models],
                          T=args.iteration)
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
