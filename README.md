# Improving Certified Defense Algorithms with Boosting

## Boosting Smoothed classifier
For this part of the implementation, we train our base models using smoothed adversarial training algorithm of Salman et, al. 2019, which is available at https://github.com/Hadisalman/smoothing-adversarial.    

### Related files/folders     
* `boosting_smooth.py`: this file contains implementation for boosting smoothed classifiers.    
* `eval_smooth_clean.py`: this file evaluate the performance of individual or ensemble of smoothed classifiers on clean data.     
* `./certify.py`: this file runs certifications for ensemble of smoothed classifiers. ( modified from [this implementation](https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/certify.py))
* `./archs`: this folder stores model architecture files we use as base models for boosting smoothed classifiers 
* `./analysis`: directly copied from [this repo](https://github.com/Hadisalman/smoothing-adversarial) for analyzing certification results of smoothed classifiers.    
* `./code`: directly copied from [this repo](https://github.com/Hadisalman/smoothing-adversarial) for running certification of smoothed classifiers.
* `./CRWON_select_certified_error.py`: this file evaluate the performance of individual or ensemble of classifiers trained with CRWON-IBP on clean andcertified accuracy. 

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
```
To evaluate ensemble of pre-trained CROWN-IBP models, run:
```
python CROWN_select_certified_error.py "eval_params:epsilon=${epsilon}" --config ${config_path} --path_prefix ${crown_ibp_pretrained_model_path} --load_ada ${path_to_adaboost_weight} --iteration {iteration}
```
