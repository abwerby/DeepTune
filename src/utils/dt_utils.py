import torch
import logging
import sys




def create_model(config, logger):
    """
    Create a model based on the provided configuration.
    model (str) : The model name to load from the hub.
    pct_to_freeze (float) : The percentage of layers to freeze.
    """
    if str(config["model"]).startswith("resnet"):
        model = torch.hub.load('pytorch/vision', config["model"].replace("_", ""), pretrained=True)
    elif str(config["model"]).startswith("dinov2"):
        model = torch.hub.load('facebookresearch/dinov2', config["model"])
    elif str(config["model"]).startswith("efficientnet"):
        model = torch.hub.load('pytorch/vision', config["model"], pretrained=True)
    elif str(config["model"]).startswith("deit"):
        model = torch.hub.load('facebookresearch/deit:main', config["model"])
    else:
        logger.error(f"Model {config['model']} not found")
        sys.exit(1)
    return model