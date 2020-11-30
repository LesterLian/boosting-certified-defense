import numpy as np
import torch
from dataset_wrapper import WeightedDataset
from ada_boost_base import AdaBoostBase


class AdaBoostTrain(AdaBoostBase):
    def gen_new_base_predictor(self, cur_round, weighted_train_dataset):
        """
        Args:
            cur_round: Current round.
            weighted_train_dataset: Weighted version of the training dataset.
        Returns:
            new_predictor: The generated new predictor.
            error: Weighted error of the new predictor on training data.
            incorrect_pred: A tensor of shape [self.num_samples] indicating the incorrect
                prediction.
        """
        new_predictor = self.base_predictor_list[cur_round]
        new_predictor.train(weighted_train_dataset)
        data_loader = torch.utils.data_loader(weighted_train_dataset, batch_size=self.num_samples)
        incorrect_pred = torch.zeros(self.num_samples)

        for _, (X, y, _) in enumerate(data_loader):
            pred = new_predictor.predict(X)
            diff = torch.abs(y - pred)
            incorrect_pred[diff.nonzero()] = 1

        error = (incorrect_pred * self.distribution).sum()

        return new_predictor, error, incorrect_pred

