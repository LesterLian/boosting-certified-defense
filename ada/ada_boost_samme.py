import torch

from ada import AdaBoostBase


class AdaBoostSamme(AdaBoostBase):
    # def __init__(self, dataset, base_predictor_list, T):
    #     super().__init__(dataset, base_predictor_list, T)

    def update_model_weight(self, error, incorrect_pred):
        if self.K is None:
            raise ValueError("Must set self.K when extending AdaBoostSamme")
        a = (self.K - 1) * torch.true_divide((1 - error), error)
        new_distributions = self.distribution * (a ** incorrect_pred)
        new_distributions = new_distributions / new_distributions.sum()
        self.distribution = new_distributions
        self.weighted_data.dataset.distribution = new_distributions
        model_weight = torch.log(a)
        print("model weight: {}".format(model_weight))
        return model_weight
