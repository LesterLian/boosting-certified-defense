import numpy as np
import torch
from tqdm import tqdm

from ada import AdaBoostBase


class AdaBoostPretrained(AdaBoostBase):
    def gen_new_base_predictor(self, cur_round, weighted_train_dataset):
        e_min = 1
        incorrect_pred_min = None
        new_predictor = None
        self.distribution.to(self.device)
        for predictor_idx, predictor in enumerate(self.base_predictor_list):
            predictor.model.eval()
            predictor.model.to(self.device)
            incorrect_pred = None
            for i, (X, y, _) in tqdm(enumerate(weighted_train_dataset), total=len(weighted_train_dataset)):
                X = X.to(self.device)
                y_pred = predictor.predict(X).argmax(dim=1).to('cpu')
                if incorrect_pred is None:
                    incorrect_pred = y_pred != y
                else:
                    incorrect_pred = torch.cat((incorrect_pred, y_pred != y), dim=0)
            error = (incorrect_pred * self.distribution).sum()
            if error < e_min:
                e_min = error
                incorrect_pred_min = incorrect_pred
                new_predictor = predictor_idx
        if e_min == 1 or new_predictor is None or incorrect_pred is None:
            raise ValueError("Didn't generate new predictor")
        print(f'choose model {new_predictor} with error {e_min}')

        return new_predictor, e_min, incorrect_pred_min

