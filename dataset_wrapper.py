import numpy as np
import torch 
from torch.utils import data


class WeightedDataset(data.Dataset):
    def __init__(self, dataset, weight):
        self.dataset = dataset
        self.weight = weight

    def __getitem__(self, index):
        elem = list(self.dataset[index])
        elem.append(self.weight[index])
        return tuple(elem)
    
    def __len__(self):
        return len(self.dataset)
