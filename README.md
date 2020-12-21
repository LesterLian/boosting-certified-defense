# Improving Certified Defense Algorithms with Boosting

This repository contains the implementations for the scholar paper authored by 
William Chen, Hong-Min Chu, Zizhen Lian, Manli Shu and Zhusheng Wang.  In this 
project, we investigate the possibility to combine boosting with adversarial 
robustness algorithm.

## Adaboost framework

The files at root are scripts using the Adaboost framework:
- general_select.py: A demo of using the framework with MNIST and Resnet

The `ada` folder contains definition of the framework
- ada_boost_base.py: `class AdaBoostBase` is the core component and constructs an Adaboost ensemble
- ada_boost_pretrained.py: `class AdaBoostPretrained` implements the weak leaner for pre-trained models.
- ada_boost_samme.py: `class AdaBoostSamme` implements the SAMME variant of Adaboost
- ada_boost_train.py: `class AdaBoostTrain`implements the weak leaner for training new models in Adaboost. (Incomplete)
- base_predictor.py: `class BasePredictor` is a wrapper class turns a `torch.nn.Module` model into a base predictor 
works with the framework and provides interface functions for adjust the model if necessary.
- dataset_wrapper.py: `class WeightedDataLoader` is a wrapper class turns a `torch.utils.data.DataLoader` or 
`torch.utils.data.Dataset` into a `DataLoader` with example weights attached.

## Boosting classifiers with linear range bounds
This part of the implementations combines the framework with pretrained models from 
[CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP).

### Files/folders   
- CROWN_select.py: Using the framework with pre-trained CROWN-IBP models.
- CROWN_train.py: Using the framework while training using CROWN-IBP. (Incomplete)

### Setups
Before running the framework with CROWN-IBP, both the environment variable `PYTHONPATH` and the 
working directory should be set to `./CROWN-IBP`.

### Commands
```bash
# General Demo
python general_select.py -T <# of rounds>

# Adaboost with CROWN-IBP
--config config/mnist_crown_large.json -T <# of rounds> [-m <path to the folder contains CROWN models>] 
[-l <file path of the saved Adaboost model>]

```

## Boosting smoothed classifier
For this part of the implementation, we train our base models using smoothed adversarial training algorithm of Salman et, al. 2019, which is available at https://github.com/Hadisalman/smoothing-adversarial.    

### Files/folders
* `boosting_smooth.py`: this file contains implementation for boosting smoothed classifiers.    
* `eval_smooth_clean.py`: this file evaluate the performance of individual or ensemble of smoothed classifiers on clean data.     
* `./certify.py`: this file runs certifications for ensemble of smoothed classifiers. ( modified from [this implementation](https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/certify.py))
* `./archs`: this folder stores model architecture files we use as base models for boosting smoothed classifiers 
* `./analysis`: directly copied from [this repo](https://github.com/Hadisalman/smoothing-adversarial) for analyzing certification results of smoothed classifiers.    
* `./code`: directly copied from [this repo](https://github.com/Hadisalman/smoothing-adversarial) for running certification of smoothed classifiers.

### Commands
To boost a set of pretrained base model with architecture `${arch}` stored at `${path}` on dataset `${dataset}` for `${T}` iterations, run:
```
python ./boosting_smooth.py --weight_dir ${path} --arch ${arch} --dataset ${dataset} --iteration ${T}
```
To evaluate the clean accuracy of a given ensemble of the above base models with ensemble model index "i1, i2, i3, ..." and ensemble weight "w1, w2, w3, ..." on dataset ${dataset}, run: 
```
python ./eval_smooth_clean.py --weight_dir ${path} --arch ${arch} --dataset ${dataset} --ensemble_weights "i1, i2, i3, ..." --ensemble_index "w1, w2, w3, ..."
```
To certify an ensemble of the above base model on the above dataset with l2 radius ${sigma}, run:
```
python ./certify.py ${dataset} ${path} ${sigma} ${output_log} --arch ${arch} --N ${N} --skip ${skip}
```
`${N}` is the number of noise samples in the Monte-Carlo certification algorithm for smoothed classifiers. `${skip}` can be set to reduce the number of test samples. For example, for CIFAR-10 with 10,000 validation samples, setting `${skip}` to 20 will only test 500 images from the validation set. This parameter can be chosen to reduce the computational cost during certification. 

## Reference

Part of the codes depends on [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP).
We appreciate very much for their great works.