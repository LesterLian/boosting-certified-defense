import argparse
import sys
import os
import time

import torch
import torchvision
from torchvision import transforms, datasets
from cifar_pretrained import AverageMeter, accuracy, MyPredictor, PretrainedSAMME


def main():
    weight_paths = [f for f in os.listdir(args.weight_dir) if os.path.isfile(os.path.join(args.weight_dir, f))]
    models = []
    print("architecture: {}".format(args.arch))
    for i in range(len(weight_paths)):
        if args.arch == 'cifar_resnet110':
            import archs.cifar_resnet as resnet
            model = resnet.resnet(depth=110, num_classes=10)
        elif args.arch == 'mnist_convnet':
            import archs.mnist_convnet as mnist_convnet
            model = mnist_convnet.mnist_convnet(num_classes=10)
        
        checkpoint = torch.load(os.path.join(args.weight_dir, weight_paths[i]))
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model = model.cuda()
        models.append(model)

    # Load dataset
    dataset_path = os.path.join('datasets', 'dataset_cache')
    print("dataset: {}".format(args.dataset))
    if args.dataset == 'cifar':
        train_data = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
        test_data = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True) 
    elif args.dataset == 'mnist':
        train_data = datasets.MNIST(dataset_path, train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST(dataset_path, train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) 

    if args.ensemble_index is None:
        for k in range(len(models)):
            eval_model = models[k]
            eval_model.eval()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for i, (input, target) in enumerate(test_loader):
                with torch.no_grad():
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    output = eval_model(input)

                    prec1, prec5 = accuracy(output, target, topk=(1, 5))
                    top1.update(prec1[0], input.size(0))
                    top5.update(prec5[0], input.size(0))
            print('Model {:s}: Clean Test Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(weight_paths[k], top1=top1, top5=top5))

    # average_weighted baseline ensemble
    ada = PretrainedSAMME(train_data,
                          # model(X) should output class number or something comparable to elements in datasets.targets
                          [MyPredictor(model) for model in models],
                          T=10)
    if args.ensemble_index is not None:
        ada.predictor_list = [int(idx) for idx in args.ensemble_index.split(",")]
        ada.predictor_weight = [torch.tensor(float(weight)) for weight in args.ensemble_weights.split(",")]
    else:
        ada.predictor_list = [i for i, _ in enumerate(models)]
        ada.predictor_weight = [torch.tensor(1/len(models)) for model in models]

    print(ada.predictor_list)
    print(ada.predictor_weight)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaboost on pre-trained CROWN-IBP models.')
    parser.add_argument('--weight_dir', type=str, 
                        help='the directory where all checkpoints stored')
    parser.add_argument('--arch', type=str, help='model architecture')
    parser.add_argument('--dataset', type=str, help='cifar/mnist')
    parser.add_argument('--ensemble_weights', type=str, default=None, help='adaboost ensemble weight')
    parser.add_argument('--ensemble_index', type=str, default=None, help='adaboost ensemble model index')

    args = parser.parse_args()
    main()
