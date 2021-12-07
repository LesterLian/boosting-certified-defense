import unittest
import torch
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from ada import AdaBoostSamme, BasePredictor


class StumpPredictor(BasePredictor):
    def train(self, weighted_data):
        self.model.fit(weighted_data.dataset.dataset.X,
                       weighted_data.dataset.dataset.y,
                       sample_weight=weighted_data.dataset.distribution)

    def predict(self, X):
        return self.model.predict(X)


class TrainSAMME(AdaBoostSamme, ClassifierMixin):
    def __init__(self, train_data, base_predictor_list, T):
        super(TrainSAMME, self).__init__(train_data, base_predictor_list, T,
                                         shuffle=False)
        # self.train_data = train_data
        self.train_data = self.weighted_data

    # Adapt to Numpy
    def predict(self, X):
        final_pred = torch.zeros((len(X), self.K)).numpy()
        for i, weight in zip(self.predictor_list, self.predictor_weight):
            cur_predictor = self.base_predictor_list[i]
            cur_weight = weight.numpy()
            if self.old:
                final_pred += cur_weight * cur_predictor.model.predict_para(X)
            else:
                classes = torch.arange(self.K).reshape(self.K, 1).numpy()
                final_pred += cur_weight * (cur_predictor.predict(X) ==
                                            classes).T

        return final_pred.argmax(axis=1)

    def gen_new_base_predictor(self, cur_round):
        # Get initialized model
        model = self.base_predictor_list[cur_round]
        # Train model

        # Convert to Ada Predictor and update
        predictor = StumpPredictor(model)
        predictor.train(self.weighted_data)
        self.base_predictor_list[cur_round] = predictor
        # Calculate Error
        incorrect_pred = None
        index = None
        error = 0
        for X, y, w, idx in tqdm(self.weighted_data,
                                 total=len(self.weighted_data)):
            X = X.cpu()
            y_pred = Tensor(predictor.predict(X))  # ndarray to Tensor
            error += ((y_pred != y) * w).sum()
            if incorrect_pred is None:
                incorrect_pred = y_pred != y
                index = idx
            else:
                incorrect_pred = torch.cat((incorrect_pred, y_pred != y),
                                           dim=0)
                index = torch.cat((index, idx), dim=0)

        return cur_round, error, incorrect_pred, index


class MyDataset(Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y
        self.classes = set(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def _set_random_states(estimator, random_state=None):
    """Set fixed random_state parameters for an estimator.

    Finds all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Parameters
    ----------
    estimator : estimator supporting get/set_params
        Estimator with potential randomness managed by random_state
        parameters.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        integers. Pass an int for reproducible output across multiple function
        calls.
        See :term:`Glossary <random_state>`.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_params()``.  ``random_state``s not controlled include
    those belonging to:

        * cross-validation splitters
        * ``scipy.stats`` rvs
    """
    # random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)

    if to_set:
        estimator.set_params(**to_set)


class MyTestCase(unittest.TestCase):
    def test_two_trump_train(self):
        # Datasets
        X, y = make_classification(n_samples=1000, n_features=4,
                                   n_informative=2, n_redundant=0,
                                   random_state=1, shuffle=True)
        dataset = MyDataset(X, y)
        for T in range(1, 101):
            with self.subTest(T=T):
                # Models
                rand = np.random.RandomState(0)
                models = [DecisionTreeClassifier(max_depth=1, random_state=0) for _
                          in range(T)]
                for m in models:
                    _set_random_states(m, rand)
                ada = TrainSAMME(dataset, models, T)
                clf = AdaBoostClassifier(n_estimators=T,
                                         random_state=0,
                                         algorithm='SAMME')
                # Train
                ada.train()
                clf.fit(X, y)
                # Eval
                ada_pred = ada.predict(Tensor([[0, 0, 0, 0]]))
                clf_pred = clf.predict([[0, 0, 0, 0]])
                ada_score = ada.score(X, y)
                clf_score = clf.score(X, y)
                # print('Ada predict:', ada_pred)
                # print('sklearn predict:', clf_pred)
                # print('Ada score', ada_score)
                # print('sklearn score', clf_score)
                self.assertEqual(ada_pred, clf_pred)
                self.assertEqual(ada_score, clf_score)


if __name__ == '__main__':
    unittest.main()
