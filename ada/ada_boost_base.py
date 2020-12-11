import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .base_predictor import BasePredictor
from .dataset_wrapper import WeightedDataLoader
from typing import Iterable, Union


class AdaBoostBase:
    def __init__(self, dataset: Union[Dataset, DataLoader], base_predictor_list: Iterable[BasePredictor],
                 T, batch_size=256, shuffle=False, num_workers=0):
        """
        Args:
            dataset: Torch.utils.data.Dataset or DataLoader. Should implement __len__() and __getitem__()
                if it's Dataset
            base_predictor_list: A list of class BasePredictor objects as base predictors.
                AdaBoost will initialize each base predictor in __init__() function.
            T: # of round for AdaBoost
        """
        if isinstance(dataset, Dataset):
            self.num_samples = dataset.__len__()
        else:
            self.num_samples = dataset.dataset.__len__()
        self.base_predictor_list = base_predictor_list
        self.T = T
        self.cur_round = 0
        # self.distribution is used in update_weight_distribution
        self.distribution = torch.Tensor([1.0 / self.num_samples] * self.num_samples)
        self.weighted_data = WeightedDataLoader(dataset, self.distribution,
                                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.K = len(self.weighted_data.dataset.dataset.classes)

        self.predictor_weight = []
        self.predictor_list = []

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def gen_new_base_predictor(self, cur_round):
        """
        Args:
            cur_round: Current round.
        Returns:
            new_predictor: The generated new predictor.
            error: Weighted error of the new predictor on training data.
            incorrect_pred: A tensor of shape [self.num_samples] indicating the incorrect
                prediction.
        """
        pass

    def update_model_weight(self, error, incorrect_pred):
        """
        Args:
            error: The weighted error for new base predictor.
            incorrect_pred: A tensor of shape [self.num_samples] indicating the incorrect
                prediction.
        Returns:
            distribution: The distribution of the new base predictor.
        """
        pass

    def train(self):
        for t in range(self.T):
            predictor, err, incorrect_pred = self.gen_new_base_predictor(t)
            weight = self.update_model_weight(err, incorrect_pred)
            self.predictor_list.append(predictor)
            self.predictor_weight.append(weight)
        # Normalize model distribution, not necessary but easy to compare
        self.predictor_weight = [i/sum(self.predictor_weight) for i in self.predictor_weight]

    def predict(self, X):
        """
        Make ensemble prediction using X, self.predictor_weight and self.predictor_list.
        Args:
            X: The input for the prediction.
        Returns:
            final_prediction: The final predicted class id.
        """
        final_pred = torch.zeros((len(X), self.K))
        X = X.to(self.device)
        for i, weight in zip(self.predictor_list, self.predictor_weight):
            cur_predictor = self.base_predictor_list[i]
            cur_weight = weight.item()
            final_pred += cur_weight * cur_predictor.model(X).to('cpu')

        return final_pred.argmax(dim=1)
