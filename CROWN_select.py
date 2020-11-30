from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, WeightedDataset
from argparser import argparser
from config import load_config, config_modelloader, config_dataloader


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10
        self.weighted_data = CROWNDataset(self.dataset, self.distribution)


class CROWNDataset(WeightedDataset):
    def __init__(self, dataset, weight):
        super(CROWNDataset, self).__init__(dataset, weight)

    def __getitem__(self, item):
        X, y = self.dataset[item]
        elem = (X.unsqueeze(0), y, self.weight[item])

        return elem


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
