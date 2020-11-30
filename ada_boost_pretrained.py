import numpy as np
import torch
from dataset_wrapper import WeightedDataset
from ada_boost_base import AdaBoostBase


class AdaBoostPretrained(AdaBoostBase):
    def gen_new_base_predictor(self, cur_round, weighted_train_loader):
        e_min = 1
        incorrect_pred_min = None
        new_predictor = None

        for predictor in self.base_predictor_list:
            predictor.model.eval()
            incorrect_pred = torch.zeros(self.num_samples)
            for _, (X, y, _) in enumerate(weighted_train_loader):
                pred = predictor.predict(X)
                diff = torch.abs(y - pred)
            incorrect_pred[diff.nonzero()] = 1
            error = (incorrect_pred * self.distribution).sum()
            if error < e_min:
                e_min = error
                incorrect_pred_min = incorrect_pred
                new_predictor = predictor

        return new_predictor, e_min, incorrect_pred_min

