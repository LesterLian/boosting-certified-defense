import numpy as np
import torch

class BasePredictor:
    def __init__(self, model, model_path=None):
        self.model_path = model_path
        self.model = model
        self.init_model_params()
    
    def init_model_params(self):
        pass
    
    # WeightedDataset provides __getitem__ that reteurns (feature, target, weight).
    def train(self, weighted_dataset):
        pass
    
    # X is of shape [N, :] where N is the # of test data for prediction.
    def predict(self, X):
        pass
