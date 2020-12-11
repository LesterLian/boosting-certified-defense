

class BasePredictor:
    def __init__(self, model, model_path=None):
        """
        Args:
            model: original model. The output of model should be a tensor contains
                score for each class.
            model_path: model will be loaded from the path if given.
        """
        self.model_path = model_path
        self.model = model
        self.init_model_params()

    def init_model_params(self):
        """
        Overwrite this method only if your model need initialization.
        """
        pass

    def train(self, weighted_data):
        """
        Train the model using given data.
        Args:
            weighted_data: WeightedDataLoader provides __getitem__ that returns (feature, target, weight).
        """
        pass

    def predict(self, X):
        """
        Make prediction for given inputs. The result should be class id.
        Args:
            X: Input to for the prediction.
        Returns:
            y: Predicted class id.
        """
        return self.model(X).argmax(dim=1)
