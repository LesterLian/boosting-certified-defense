import numpy as np
import torch
from datasets_wrapper import WeightedDataset

class AdaBoostSamme(AdaBoostBase):
    def __init__(self, dataset, base_predictor_list, T):
        weighted_train_dataset: Weighted version of the training dataset.
        super().__init__(dataset, base_predictor_list, T)
        self.K = 10

    def update_weight_distribution(self, error, incorrect_pred):
    """
    Args:
        error: The weighted error for new base predictor.
        incorrect_pred: A tensor of shape [self.num_samples] indicating the incorrect
            prediction.
    Returns:
        weight: The weight of the new base predictor.
        distribution: The new distribution of training data.
    """
        a = (self.K - 1) * torch.true_divide((1 - error), error
        new_distributions = self.distributions * (a ** incorrect_pred)
        return torch.log(a), new_distributions
