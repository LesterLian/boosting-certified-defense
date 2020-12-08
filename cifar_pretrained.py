import argparse
import sys
import os
import time

import torch
import torchvision
from torchvision import transforms, datasets

from ada import AdaBoostPretrained, AdaBoostSamme, BasePredictor, WeightedDataset


class PretrainedSAMME(AdaBoostPretrained, AdaBoostSamme):
    def __init__(self, dataset, base_predictor_list, T):
        super(PretrainedSAMME, self).__init__(dataset, base_predictor_list, T)
        # TODO recognize number of class
        self.K = 10


# Overwrite predictor.predict() because the input and output dimension doesn't match for MNIST and ResNet
class MyPredictor(BasePredictor):
    def __init__(self, model):
        super(MyPredictor, self).__init__(model)

    def predict(self, X):
        output = self.model(X.unsqueeze(0))
        return torch.argmax(output)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    # Load pre-trained models
    weight_paths = [f for f in os.listdir(args.weight_dir) if os.path.isfile(os.path.join(args.weight_dir, f))]
    models = []
    for i in range(len(weight_paths)):
        if args.arch == 'cifar_resnet110':
            import archs.cifar_resnet as resnet
            model = resnet.resnet(depth=110, num_classes=10)
            checkpoint = torch.load(os.path.join(args.weight_dir, weight_paths[i]))
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            models.append(model)

    # Load dataset
    dataset_path = os.path.join('datasets', 'dataset_cache')
    train_data = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    test_data = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    # Boost
    ada = PretrainedSAMME(train_data,
                          # model(X) should output class number or something comparable to elements in datasets.targets
                          [MyPredictor(model) for model in models],
                          T=args.iteration)
    ada.train()

    ### test on clean data
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = ada.predict(input)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
    print('Clean Test Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    print(ada.predictor_list)
    print(ada.predictor_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--weight_dir', type=str, 
                        help='the directory where all checkpoints stored')
    parser.add_argument('--arch', type=str, 
                        help='model architecture')
    parser.add_argument('--iteration', '-T', type=int, default=10, 
                        help='the maximum number of running Adaboost')

    args = parser.parse_args()
    main()
