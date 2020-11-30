from config import load_config, config_modelloader, config_dataloader
from argparser import argparser
import torch

from ada_boost_pretrained import AdaBoostPretrained
from ada_boost_samme import AdaBoostSamme
from base_predictor import BasePredictor


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10


def main():
    config = load_config(args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain=True)

    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_eval_config["loader_params"])
    # Boost
    # TODO T as argument
    ada = PretrainedSAMME(train_data.dataset,
                          [BasePredictor(model) for model in models],
                          T=10)
    ada.train()

    print(ada.predictor_list)
    print(ada.predictor_weight)


if __name__ == '__main__':
    args = argparser()
    main()
