from torch.utils.data import Dataset, DataLoader
from typing import Union


class WeightedDataLoader(DataLoader):
    def __init__(self, dataset: Union[Dataset, DataLoader], weight, batch_size=256, shuffle=False, num_workers=0):
        if isinstance(dataset, Dataset):
            dataset = WeightedDataset(dataset, weight)
            super(WeightedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=num_workers)
        else:
            self.__dict__ = dataset.__dict__.copy()
            self.__dict__['_DataLoader__initialized'] = False
            self.dataset: WeightedDataset = WeightedDataset(dataset.dataset, weight)
        # self.dataset = dataset
        # self.weight = weight


class WeightedDataset(Dataset):
    def __init__(self, dataset, weight):
        self.dataset = dataset
        self.weight = weight

    def __getitem__(self, index):
        elem = list(self.dataset[index])
        elem.append(self.weight[index])
        return tuple(elem)

    def __len__(self):
        return len(self.dataset)

    def update_weight(self, weight):
        self.weight = weight
