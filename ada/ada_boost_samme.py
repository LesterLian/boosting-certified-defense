import torch

from ada import AdaBoostBase


class AdaBoostSamme(AdaBoostBase):
    # def __init__(self, dataset, base_predictor_list, T):
    #     super().__init__(dataset, base_predictor_list, T)

    def update_model_weight(self, error, incorrect_pred, index):
        if self.K is None:
            raise ValueError("Must set self.K when extending AdaBoostSamme")
        a = (self.K - 1) * torch.true_divide((1 - error), error)
        new_distributions = self.distribution[index] * (a ** incorrect_pred)
        new_distributions = new_distributions / new_distributions.sum()
        self.distribution[index] = new_distributions
        # self.weighted_data.dataset.distribution = new_distributions
        print(f"a: {a.item()}")

        return torch.log(a)
