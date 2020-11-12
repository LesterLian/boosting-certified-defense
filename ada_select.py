import torch
from config import load_config, config_modelloader, config_dataloader
from argparser import argparser
import copy


def SAMME(model_packs, distribution):
    # Choose model based on distribution
    model_id, e, incorrect, distribution = select_model(model_packs, distribution)
    # Stop if weak model too strong
    if e > 1/2:
        return None
    # Update distribution
    # TODO
    K = 10
    a = (K-1) * torch.true_divide((1-e), e)
    distribution = distribution * (a ** incorrect)

    return distribution, a


def M1(model_packs, distribution):
    # Choose model based on distribution
    model_id, e, incorrect, distribution = select_model(model_packs, distribution)
    # Stop if weak model too strong
    if e > 1/2:
        return None
    # Update distribution
    a = torch.true_divide((1-e), e)
    distribution = distribution * (a ** incorrect)

    return distribution, torch.log(a)


def select_model(model_packs, distribution):
    print(f"Choose model for {distribution}")
    e_min = None

    for model_id, (model, data) in model_packs.items():
        # Initialize varaibles
        model.eval()
        model = model.cuda()
#         with torch.set_grad_enabled(False):
        batch_size = data.batch_size
        runs = 0
        incorrect = torch.zeros_like(distribution)
        # Compute error for each batch
        for i, (X, y) in enumerate(data):
            X = X.cuda()
            y = y.cuda()
            output = model(X)
            y_pred = torch.argmax(output, 1)
            incorrect[batch_size*i:batch_size*i+len(X)] += y_pred != y
            runs += 1
        # Update the minimum error and model
        # print(f"{model_id}: {torch.dot(distribution, incorrect)}")
        if e_min is None or torch.dot(distribution, incorrect) < e_min:
            e_min = torch.dot(distribution, incorrect)
            id_min = model_id
            incorrect_min = incorrect
            print(f"Best model: {model_id}  error: {e_min}")

    return id_min, e_min, incorrect_min, distribution


def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain=True)
    model_packs = {}
    distribution = None
    model_weights = []
    T = 10  # regulation

    # Models should use same training set
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        # Initialize Config
        # make a copy of global training config, and update per-model config
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])

        # Initialize Model
        #         model = model.cuda()
        # Initialize Data
        train_data, test_data = config_dataloader(config, **eval_config["loader_params"])
        n = len(train_data.dataset)
        model_packs[model_id] = (model, train_data)
    for i in range(T):
        # Update weights
        distribution = torch.tensor([1 / n] * n).cuda()
        result = M1(model_packs, distribution)
        if result is None:
            break
        distribution, a = result
        model_weights.append(a)

    print(model_weights)


if __name__ == "__main__":
    args = argparser()
    main(args)
