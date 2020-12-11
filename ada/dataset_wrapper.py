from torch.utils.data import Dataset, DataLoader
from typing import Union


class WeightedDataLoader(DataLoader):
    def __init__(self, dataset: Union[Dataset, DataLoader], distribution, batch_size=256, shuffle=False, num_workers=0):
        """
        Args:
            dataset: Torch.utils.data.Dataset or DataLoader.
            distribution: distribution.
            batch_size: # of round for AdaBoost
            shuffle: whether construct shuffled DataLoader
            num_workers: # of workers for multiprocessing
        """
        if isinstance(dataset, Dataset):
            dataset = WeightedDataset(dataset, distribution)
            super(WeightedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=num_workers)
        else:
            self.__dict__ = dataset.__dict__.copy()
            self.__dict__['_DataLoader__initialized'] = False
            self.dataset: WeightedDataset = WeightedDataset(dataset.dataset, distribution)
        # self.dataset = dataset
        # self.distribution = distribution


class WeightedDataset(Dataset):
    def __init__(self, dataset, distribution):
        self.dataset = dataset
        self.distribution = distribution

    def __getitem__(self, index):
        elem = list(self.dataset[index])
        elem.append(self.distribution[index])
        return tuple(elem)

    def __len__(self):
        return len(self.dataset)
