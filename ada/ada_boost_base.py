import numpy as np
import torch
from .dataset_wrapper import WeightedDataset


class AdaBoostBase:
    def __init__(self, dataset, base_predictor_list, T):
        """
        Args:
            dataset: Torch dataset. Should implement __len__() and __getitem__()
            base_predictor_list: A list of base predictors. AdaBoost will
                initialize each base predictor in __init__() function.
            T: # of round for AdaBoost
        """

        self.num_samples = dataset.__len__()
        self.base_predictor_list = base_predictor_list
        self.T = T
        self.K = len(dataset.classes)
        self.cur_round = 0
        # self.dataset = dataset
        self.distribution = torch.Tensor([1.0 / self.num_samples] * self.num_samples)
        self.weighted_data = WeightedDataset(dataset, self.distribution)

        self.predictor_weight = []
        self.predictor_list = []

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

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
        pass
    
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
        pass

    def train(self):
        cur_round = 0
        for predictor in self.base_predictor_list:
            predictor.init_model_params()

        for t in range(self.T):
            print(f"running iter {t}")
            predictor, err, incorrect_pred = self.gen_new_base_predictor(cur_round, self.weighted_data)
            weight, self.distribution = self.update_weight_distribution(err, incorrect_pred)
            self.predictor_list.append(predictor)
            self.predictor_weight.append(weight)
            cur_round += 1
        self.predictor_weight = [i/sum(self.predictor_weight) for i in self.predictor_weight]

    def predict(self, X):
        final_pred = torch.zeros((len(X), self.K))
        X = X.to(self.device)
        for i in range(len(self.predictor_list)):
            cur_predictor = self.predictor_list[i]
            cur_weight = self.predictor_weight[i].item()
            # if final_pred is None:
            #     final_pred = cur_weight * cur_predictor.predict(X)
            # else:
            #     final_pred += cur_weight * cur_predictor.predict(X)
            class_pred = cur_predictor.predict(X).to('cpu')
            final_pred.scatter_(1, class_pred.view(-1, 1), cur_weight, reduce='add')

        return final_pred.argmax(1)
