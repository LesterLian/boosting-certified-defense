import numpy as np
import torch
from tqdm import tqdm

from ada import AdaBoostBase


class AdaBoostPretrained(AdaBoostBase):
    def gen_new_base_predictor(self, cur_round, weighted_train_dataset):
        e_min = 1
        incorrect_pred_min = None
        new_predictor = None
        for predictor in self.base_predictor_list:
            predictor.model.eval()
            predictor.model.to(self.device)
            incorrect_pred = torch.zeros(self.num_samples)
            for i, (X, y, _) in tqdm(enumerate(weighted_train_dataset), total=len(weighted_train_dataset)):
                X = X.to(self.device)
                y_pred = predictor.predict(X)
                incorrect_pred[i] = 1 if y_pred != y else 0
            # incorrect_pred[torch.nonzero(diff, as_tuple=False)] = 1
            error = (incorrect_pred * self.distribution).sum()
            if error < e_min:
                e_min = error
                incorrect_pred_min = incorrect_pred
                new_predictor = predictor
        if e_min == 1 or new_predictor is None or incorrect_pred is None:
            raise ValueError("Didn't generate new predictor")

        return new_predictor, e_min, incorrect_pred_min

