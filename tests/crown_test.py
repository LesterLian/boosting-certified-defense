import copy
import unittest
from pathlib import Path

import torch


class MyTestCase(unittest.TestCase):
    def test_two_crown_train(self):
        model_path = '../CROWN-IBP/mnist_crown_large'
        models = list(Path(model_path).glob('*.pth'))
        self.assertTrue(len(models) > 1, '''first run: 
        train.py --config config/mnist_crown_large_test.json --model_subset 0 
        train.py --config config/mnist_crown_large_test.json --model_subset 1''')

        model_1 = torch.load(models[0])['state_dict']
        model_2 = torch.load(models[1])['state_dict']
        for (k1, t1), (k2, t2) in zip(model_1.items(), model_2.items()):
            if not torch.equal(t1, t2):
                self.fail(f'Tensor not match at state_dict["{k1}"]')


if __name__ == '__main__':
    unittest.main()
