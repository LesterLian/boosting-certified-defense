import unittest

from torch import Tensor, arange

from torch.utils.data import Dataset

from ada.dataset_wrapper import WeightedDataset, WeightedDataLoader
from config import config_dataloader, load_config


class BasicDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, item):
        return Tensor([item])


dataset = BasicDataset()


class BasicTest(unittest.TestCase):
    def test_dataset(self):
        weighted_dataset: WeightedDataset = WeightedDataset(dataset, arange(100))
        for example, distribution, idx in weighted_dataset:
            self.assertEqual(example, distribution)

    # def test_dataset_shuffle(self):
    #     self.assertEqual(True, False)

    def test_dataloader(self):
        dataloader = WeightedDataLoader(dataset, arange(100), batch_size=2)
        for example, distribution, idx in dataloader:
            self.assertTrue((example == distribution).all())
            self.assertTrue((example == idx).all())

    def test_dataloader_shuffle(self):
        dataloader = WeightedDataLoader(dataset, arange(100), batch_size=2, shuffle=True)
        for example, distribution, idx in dataloader:
            self.assertTrue((example == distribution).all())
            self.assertTrue((example == idx).all())
        # This shows 1) the batch is random and 2) example = distribution = index
        print(f'test_dataloader_shuffle:\nlast shuffle: {example} {distribution}')

    def test_mnist_shuffle(self):
        config = {'dataset': 'mnist'}
        loader_params = {'batch_size': 256, 'shuffle_train': True, 'test_batch_size': 256}
        train_data, test_data = config_dataloader(config, **loader_params)
        # Use train data because train loader is shuffled but test loader is not
        dataloader = WeightedDataLoader(train_data, arange(len(train_data.dataset)), batch_size=2, shuffle=True)
        for example, label, distribution, idx in dataloader:
            self.assertTrue((distribution == idx).all())
        # This shows 1) the batch is random and 2) distribution = index
        print(f'test_dataloader_shuffle:\nlast shuffle: {example.sum()} \n{distribution} \n{idx}')


# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
