# Improving Certified Defense Algorithms with Boosting

This repository contains the implementations for the scholar paper authored by 
William Chen, Hong-Min Chu, Zizhen Lian, Manli Shu and Zhusheng Wang.  In this 
project, we investigate the possibility to combine boosting with adversarial 
robustness algorithm.

## Adaboost framework

The files at root are scripts using the Adaboost framework:
- `general_select.py`: A demo of using the framework with MNIST and Resnet

The `ada` folder contains definition of the framework
- `ada_boost_base.py`: `class AdaBoostBase` is the core component and constructs an Adaboost ensemble
- `ada_boost_pretrained.py`: `class AdaBoostPretrained` implements the weak leaner for pre-trained models.
- `ada_boost_samme.py`: `class AdaBoostSamme` implements the SAMME variant of Adaboost
- `ada_boost_train.py`: `class AdaBoostTrain`implements the weak leaner for training new models in Adaboost. (Incomplete)
- `base_predictor.py`: `class BasePredictor` is a wrapper class turns a `torch.nn.Module` model into a base predictor 
works with the framework and provides interface functions for adjust the model if necessary.
- `dataset_wrapper.py`: `class WeightedDataLoader` is a wrapper class turns a `torch.utils.data.DataLoader` or 
`torch.utils.data.Dataset` into a `DataLoader` with example weights attached.

## Boosting classifiers with linear range bounds
This part of the implementations combines the framework with pretrained models from 
[CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP).

### Files/folders   
- `CROWN_select.py`: Using the framework with pre-trained CROWN-IBP models.
- `CROWN_train.py`: Using the framework while training using CROWN-IBP. (Incomplete)
- `./CRWON_select_certified_error.py`: this file evaluate the performance of individual or ensemble of classifiers trained with CRWON-IBP on clean andcertified accuracy. 


### Setups
Before running the framework with CROWN-IBP, both the environment variable `PYTHONPATH` and the 
working directory should be set to `./CROWN-IBP`.

### Commands
With config at `${config_path}`, pre-trained CROWN-IBP models at `${crown_ibp_pretrained_model_path}` 
running `${iteration}` iterations of Adaboost, and previous saved Adaboost model at `${path_to_saved_adaboost_model}` 
and `${epsilon}` for evaluation, we use following commands.

General Demo
```bash
python general_select.py -T ${iteration}
```
To apply Adaboost framework on CROWN-IBP, run:
```bash
python CROWN_select.py --config ${config_path} -T ${iteration} [-m ${crown_ibp_pretrained_model_path}] 
[--load_ada ${path_to_saved_adaboost_model}]
```
To evaluate ensemble of pre-trained CROWN-IBP models, run:
```bash
python CROWN_select_certified_error.py "eval_params:epsilon=${epsilon}" --config ${config_path} 
--path_prefix ${crown_ibp_pretrained_model_path} --load_ada ${path_to_saved_adaboost_model} --iteration {iteration}
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
Part of our code is completed base on the implementations from 
[CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP) and 
[Randomized Smoothing](https://github.com/Hadisalman/smoothing-adversarial).    
We appreciate very much for their great works.