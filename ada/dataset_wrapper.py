from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from typing import Union


class WeightedDataLoader(DataLoader):
    def __init__(self, dataset: Union[Dataset, DataLoader], distribution, batch_size=256, shuffle=False, num_workers=4):
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
            if shuffle and isinstance(self.sampler, RandomSampler):
                self.sampler = RandomSampler(self.dataset, generator=self.generator)
            elif not shuffle:
                self.sampler = SequentialSampler(self.dataset)
            self.batch_sampler.sampler = self.sampler
            # self.dataset: WeightedDataset = WeightedDataset(dataset.dataset, distribution)
            # super(WeightedDataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
            #                                          shuffle=shuffle, num_workers=num_workers)
            # sampler = self.sampler
            # batch_sampler = self.batch_sampler
            # dataset_tmp = self.dataset
            # self.__dict__ = dataset.__dict__.copy()
            # self.__dict__['_DataLoader__initialized'] = False
            # self.sampler = sampler
            # self.batch_sampler = batch_sampler
            # self.dataset = dataset_tmp


class WeightedDataset(Dataset):
    def __init__(self, dataset, distribution):
        self.dataset = dataset
        self.distribution = distribution

    def __getitem__(self, index):
        return (*self.dataset[index], self.distribution[index], index)

    def __len__(self):
        return len(self.dataset)
