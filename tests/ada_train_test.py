import copy
import unittest
import torch


class MyTestCase(unittest.TestCase):
    def test_two_crown_train(self):
        for name in ["cnn_3layer_fixed_kernel_3_width_8_best.pth", "cnn_4layer_linear_512_width_4_best.pth", "cnn_4layer_linear_512_width_16_best.pth"]:
            model_1 = torch.load(f"../CROWN-IBP/mnist_crown_large/1/{name}")
            model_2 = torch.load(f"../CROWN-IBP/mnist_crown_large/2/{name}")
            for (k1, t1), (k2, t2) in zip(model_1['state_dict'].items(), model_2['state_dict'].items()):
                if not torch.equal(t1, t2):
                    self.fail(f'Tensor not match at state_dict["{k1}"]')

    def test_ada_train(self):
        for name in ["cnn_4layer_linear_512_width_4_best.pth"]:
            model_1 = torch.load(f"../CROWN-IBP/mnist_crown_large/1/{name}")
            model_2 = torch.load(f"../CROWN-IBP/mnist_crown_large/ada/{name}")
            for (k1, t1), (k2, t2) in zip(model_1['state_dict'].items(), model_2['state_dict'].items()):
                if not torch.equal(t1, t2):
                    self.fail(f'Tensor not match at state_dict["{k1}"]')
