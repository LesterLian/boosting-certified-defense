import numpy as np
import torch
from tqdm import tqdm

from ada import AdaBoostBase


class AdaBoostPretrained(AdaBoostBase):
    def gen_new_base_predictor(self, cur_round):
        e_min = 1
        incorrect_pred_min = None
        index_min = None
        new_predictor = None
        # self.distribution.to(self.device)
        print(f"running round {cur_round}")
        print(f"Max distribution: {self.distribution.max()}")
        for predictor_idx, predictor in enumerate(self.base_predictor_list):
            predictor.model.eval()
            predictor.model.to(self.device)
            incorrect_pred = None
            index = None
            error = 0
            for X, y, w, idx in tqdm(self.weighted_data, total=len(self.weighted_data)):
                X = X.to(self.device)
                y_pred = predictor.predict(X).to('cpu')
                error += ((y_pred != y) * w).sum()
                if incorrect_pred is None:
                    incorrect_pred = y_pred != y
                    index = idx
                else:
                    incorrect_pred = torch.cat((incorrect_pred, y_pred != y), dim=0)
                    index = torch.cat((index, idx), dim=0)
            # error = (incorrect_pred * self.distribution).sum()
            if error < e_min:
                e_min = error
                incorrect_pred_min = incorrect_pred
                index_min = index
                new_predictor = predictor_idx
            print(f'model{predictor_idx}: error {error.item():.6f} incorrect {incorrect_pred.sum().item()}')
        if e_min == 1 or new_predictor is None or incorrect_pred is None:
            raise ValueError("Didn't generate new predictor")
        print(f'choose model {new_predictor} with error {e_min}')

        return new_predictor, e_min, incorrect_pred_min, index_min

