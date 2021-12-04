import sys

import torch
from torch.utils.data import Dataset, DataLoader
from .base_predictor import BasePredictor
from .dataset_wrapper import WeightedDataLoader
from .modules import log
from typing import Iterable, Union


class AdaBoostBase:
    def __init__(self, dataset: Union[Dataset, DataLoader],
                 base_predictor_list: Iterable[BasePredictor], T,
                 batch_size=256, shuffle=False, num_workers=0, old=False,
                 test_data=None):
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
        self.test_data = test_data
        self.do_val = False if self.test_data is None else True
        self.K = len(self.weighted_data.dataset.dataset.classes)
        self.classes = torch.arange(self.K).reshape(1, self.K).cuda()

        self.predictor_weight = []
        self.predictor_list = []

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.old = old

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

    def update_model_weight(self, error, incorrect_pred, index):
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
            predictor, err, incorrect_pred, index = self.gen_new_base_predictor(t)
            weight = self.update_model_weight(err, incorrect_pred, index)
            self.predictor_list.append(predictor)
            self.predictor_weight.append(weight)
            if self.do_val:
                incorrect = 0
                for X, y in self.test_data:
                    y_pred = self.predict(X).detach()
                    incorrect += (y_pred != y).sum()
                log({'Clean Error', incorrect / len(
                    self.test_data.dataset)}, t)
        # Normalize model distribution, not necessary but easy to compare
        self.predictor_weight = [i/sum(self.predictor_weight) for i in self.predictor_weight]

    def predict(self, X):
        """
        Make ensemble prediction using X, self.predictor_weight and self.predictor_list.
        Args:
            X: The input for the prediction.
        Returns:
            final_prediction: The final predicted probability of each class id.
        """
        final_pred = torch.zeros((len(X), self.K)).cuda()
        X = X.to(self.device)
        for i, weight in zip(self.predictor_list, self.predictor_weight):
            cur_predictor = self.base_predictor_list[i]
            cur_predictor.model.eval()
            cur_weight = weight.cuda()
            if self.old:
                print('Warning: Using Old predict method', file=sys.stderr)
                final_pred += cur_weight * cur_predictor.model(X)
            else:
                probas = torch.tensor(cur_predictor.predict(X))
                probas = probas.reshape(-1, 1) == self.classes
                final_pred += cur_weight * probas

        return final_pred
