## Selecting Pre-trained Models

```bash
# General Demo
python general_select.py -T 1
```

### Change log

Dec 11, 2020
- AdaBoostBase can take Dataset or DataLoader and always construct DataLoader under the hood and store at self.weighted_data.
- AdaBoostBase takes arguments for DataLoader
- BasePredictor.predict(X) should return class id. Use BasePredictor.model(X) for raw output.
- BasePredictor.predict() only needs to be overwritten if the output of model(X) doesn't match number of classes.
- Number of classes K is read from Dateset.classes; I'm not sure if all Dataset has that attribute.