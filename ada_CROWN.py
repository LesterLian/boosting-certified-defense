from config import load_config, config_modelloader, config_dataloader
from argparser import argparser

from ada_wrapper import SAMME

def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]
    models, model_names = config_modelloader(config, load_pretrain=True)
    T = 10  # regulation

    # Initialize Data
    train_data, test_data = config_dataloader(config, **global_eval_config["loader_params"])
    n = len(train_data.dataset)
    # Boost
    ada = SAMME(models, T=T, K=10, n=n)
    model_weights = ada.forward(train_data)
    print(model_weights)

if __name__ == "__main__":
    args = argparser()
    main(args)