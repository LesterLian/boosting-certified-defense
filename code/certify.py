# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from torchvision import transforms, datasets
from code.core import Smooth
import torch


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", help="which dataset")
parser.add_argument("weight_dir", type=str, help="path to the directory where base models are stored")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--arch', type=str, help='architecture')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--ensemble_weights", type=str, default=None, help='path to saved set of weights')
parser.add_argument('--ensemble_index', type=str, default=None, help='adaboost ensemble model index')
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifiers

    paths = [f for f in os.listdir(args.weight_dir) if os.path.isfile(os.path.join(args.weight_dir, f))]

    # load weights here, maybe using args.weights
    # for now, hard code the weights to be [0.25, 0.25, 0.25, 0.25]
    # weights: List[float]
    if args.ensemble_index is not None:
        weights = [float(weight) for weight in args.ensemble_weights.split(",")]
        index = [int(idx) for idx in args.ensemble_index.split(",")]
    else:
        weights = [1.0/len(paths) for path in paths]
        index = [i for i, _ in enumerate(paths)]

    base_classifiers = []
    for i in index:
        path = paths[i]
        if torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(args.weight_dir, path))
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if args.arch == 'cifar_resnet110':
            import archs.cifar_resnet as resnet
            base_classifier = resnet.resnet(depth=110, num_classes=10)
        elif args.arch == 'mnist_convnet':
            import archs.mnist_convnet as mnist_convnet
            base_classifier = mnist_convnet.mnist_convnet(num_classes=10)
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifier = base_classifier.cuda()
        base_classifiers.append(base_classifier)

    # if torch.cuda.is_available():
    #     checkpoint = torch.load(args.base_classifier)
    # else:
    #     checkpoint = torch.load(args.base_classifier, map_location=torch.device('cpu'))
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    # base_classifier.load_state_dict(checkpoint['state_dict'])
    #
    # checkpoint = torch.load(args.base_classifier, map_location=torch.device('cpu')) #remove map_location keyword arg to run on cuda
    # base_classifier2 = get_architecture(checkpoint["arch"], args.dataset)
    # base_classifier2.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifiers, weights, 10, args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    # Load dataset
    dataset_path = os.path.join('datasets', 'dataset_cache')
    print("dataset: {}".format(args.dataset))
    if args.dataset == 'cifar':
        dataset = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'mnist':
        dataset = datasets.MNIST(dataset_path, train=False, download=True, transform=transforms.ToTensor())
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        if torch.cuda.is_available():
            x = x.cuda()
        else:
            x = x
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
