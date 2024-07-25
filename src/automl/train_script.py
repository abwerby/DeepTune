import logging
import time
import torch
from typing import Optional, Dict, Any
from argparse import ArgumentParser
import torchvision

from syne_tune import Reporter
from syne_tune.config_space import randint
from syne_tune.utils import add_config_json_to_argparse, load_config_json
from deeptune import DeepTune


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    # Append required argument(s):
    add_config_json_to_argparse(parser)
    args, _ = parser.parse_known_args()
    # Loads config JSON and merges with ``args``
    config = load_config_json(vars(args))
    
    root.info("Starting training...")
    root.info("Config: %s", config)
    
    dataset = torchvision.datasets.CIFAR10("data", train=True, download=True)

    # give the device to the deeptune based on what is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root.info(f"Device: {device}")
    root.info(f"Number of epochs: {config['epochs']}")
    
    result = DeepTune(
        seed=42,
        device=device,
        logger=root,
    ).fit(
        dataset_class=dataset,
        ConfigSpace=config,
        epochs=config["epochs"],
    )
    
    result = {
        "loss": result[0],
        "cost": result[1],
    }
    root.info("Training finished.")
    report = Reporter()
    report(**result)
    
