## Selecting Pre-trained Models

```bash
# General Demo
python general_select.py -T 1
```

<<<<<<< HEAD
### Change log

Dec 11, 2020
- AdaBoostBase can take Dataset or DataLoader and always construct DataLoader under the hood and store at self.weighted_data.
- AdaBoostBase takes arguments for DataLoader
- BasePredictor.predict(X) should return class id. Use BasePredictor.model(X) for raw output.
- BasePredictor.predict() only needs to be overwritten if the output of model(X) doesn't match number of classes.
- Number of classes K is read from Dateset.classes; I'm not sure if all Dataset has that attribute.
=======


I found it helpful to read the README's from the following two github repo's to use the right commands:
1) https://github.com/Hadisalman/smoothing-adversarial
2) https://github.com/locuslab/smoothing

3) To use the right command, it might help to look at the parser code in code/certify.py, lines (13-26).


I made several changes, to handle a weighted ensemble of models.

1)  when running code/certify.py, the command-line argument specifying the models should be
    a comma-separated list of paths
2)  since I didn't know how you were saving the weights, the weights are currently hard-coded to [0.25, 0.25, 0.25, 0.25]
    because I was testing out the average of the four base classifiers.
    In order to provide the weights using a file containing the model weights, I will need to change code/certify.py,
    at approximately lines (42-45).
3)  I added to the parser the "--weights" command-option, which allows a path to the file of saved weights to be given.
    This will be used in the code change for the previous item.

An example command to run is:

model="cifar_res110/checkpoint_0.pth.tar,cifar_res110/checkpoint_1.pth.tar,cifar_res110/checkpoint_2.pth.tar,cifar_res110/checkpoint_3.pth.tar"
output="data/certification_output"
python code/certify.py cifar10 $model 0.12 $output --skip 5000 --batch 400 --N 100

The optional command-line arguments I used here were for debugging. They can be left out when getting the actual results.
Note the destination file and sigma (here, it's 0.12) can be changed with running a different experiment. 
>>>>>>> cb039d8088cbbcaad4b7419da0c6af0d3a9767da
