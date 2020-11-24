import torch
from torch import nn

from typing import List, Union


class SAMME(nn.Module):
    def __init__(self, base_model: Union[nn.Module , List[nn.Module]],
                 T=10, K=0, n=0, distributions: torch.tensor = None):
        super(SAMME, self).__init__()
        self.T = T
        self.model_weights = []
        self.data = None
        if isinstance(base_model, list):
            self.train = False
            self.pretrained_models = base_model
        else:
            self.train = True
            self.base_model = base_model
        if K == 0:
            m = list(base_model.modules())[-1]
            # TODO this may not exist
            self.K = m.out_features
        else:
            self.K = K
        if n == 0 and distributions is None:
            raise ValueError("Must provide data size or example distributions")
        elif n == 0:
            self.distributions = distributions.cuda()
        else:
            self.distributions = torch.tensor([1 / n] * n).cuda()
        # TODO maybe add some stats

    def forward(self, data):  # TODO do we really want forward?
        self.data = data
        for i in range(self.T):
            # Get a model from the weak learner
            if self.train:
                model, e, incorrect = self.train_model()
            else:
                model, e, incorrect = self.select_model()
            # Stop if weak model too strong
            if e > 1 / 2:
                return None
            # Update distribution
            a = (self.K - 1) * torch.true_divide((1 - e), e)
            self.distributions = self.distributions * (a ** incorrect)
            self.model_weights.append(torch.log(a))

        return self.model_weights

    def select_model(self):
        print(f"Choose model for {self.distributions}")
        e_min = None
        model_min = None

        for model in self.pretrained_models:
            # Initialize varaibles
            model.eval()
            model = model.cuda()
            #         with torch.set_grad_enabled(False):
            # TODO non-batch or non-DataLoader case
            batch_size = self.data.batch_size
            runs = 0
            incorrect = torch.zeros_like(self.distributions).cuda()
            # Compute error for each batch
            for i, (X, y) in enumerate(self.data):
                X = X.cuda()
                y = y.cuda()
                output = model(X)
                y_pred = torch.argmax(output, 1)
                incorrect[batch_size * i:batch_size * i + len(X)] += y_pred != y
                runs += 1
            # Update the minimum error and model
            # print(f"{model_id}: {torch.dot(distribution, incorrect)}")
            if e_min is None or torch.dot(self.distributions, incorrect) < e_min:
                e_min = torch.dot(self.distributions, incorrect)
                model_min = model
                incorrect_min = incorrect
                print(f"Best model: {model}\n  error: {e_min}")

        return model_min, e_min, incorrect_min

    def train_model(self):
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.base_model.train()
        for i, (X, y) in enumerate(self.data):
            outputs = self.base_model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        # Evaluation
        self.base_model.eval()
        incorrect = torch.zeros_like(self.distributions)
        batch_size = self.data.batch_size
        with torch.no_grad():
            for i, (X, y) in enumerate(self.test_data):
                outputs = self.base_model(X)

                y_pred = torch.argmax(outputs, 1)
                incorrect[batch_size * i:batch_size * i + len(X)] += y_pred != y
        e = torch.dot(self.distribution, incorrect)

        return self.base_model, e, incorrect


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



