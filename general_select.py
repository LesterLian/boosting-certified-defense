import argparse
import sys

import torch
import torchvision

from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, WeightedDataLoader


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10


# Overwrite predictor.predict() because the input and output dimension doesn't match for MNIST and ResNet
class MyPredictor(BasePredictor):
    def __init__(self, model):
        super(MyPredictor, self).__init__(model)

    def predict(self, X):
        output = self.model(X.unsqueeze(0))
        return torch.argmax(output)


def main():
    # Load pre-trained models

    models = [torchvision.models.resnet18(pretrained=True)]
    # Load dataset
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=trans)
    # Boost
    T = 10 if args.iteration is None else args.iteration
    ada = PretrainedSAMME(train_data,
                          # model(X) should output class number or something comparable to elements in datasets.targets
                          [MyPredictor(model) for model in models],
                          T=T)
    ada.train()

    print(ada.predictor_list)
    print(ada.predictor_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--iteration', '-T', type=int,
                        help='the maximum number of running Adaboost')

    args = parser.parse_args()
    main()
