import torch

from ada import AdaBoostBase


class AdaBoostSamme(AdaBoostBase):
    def __init__(self, dataset, base_predictor_list, T):
        # weighted_train_dataset: Weighted version of the training dataset.
        super().__init__(dataset, base_predictor_list, T)

    def update_weight_distribution(self, error, incorrect_pred):
        if self.K is None:
            raise ValueError("Must set self.K when extending AdaBoostSamme")
        a = (self.K - 1) * torch.true_divide((1 - error), error)
        new_distributions = self.distribution * (a ** incorrect_pred)
        new_distributions = new_distributions / new_distributions.sum()

        return torch.log(a), new_distributions
