import logging
import time
import torch
from typing import Optional, Dict, Any
from argparse import ArgumentParser
import torchvision
from torchvision import transforms

from syne_tune import Reporter
from syne_tune.config_space import randint
from syne_tune.utils import add_config_json_to_argparse, load_config_json

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from tune.deeptune import DeepTune
from tune.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from utils.deeptune_utils import calculate_mean_std


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
        
    # define base transformation based on the dataset
    
    if config["dataset_name"] == "FashionDataset":
        Dataset = FashionDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )
    elif config["dataset_name"] == "FlowersDataset":
        Dataset = FlowersDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
        ]
    )
    elif config["dataset_name"] == "EmotionsDataset":
        Dataset = EmotionsDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((42, 42)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )
    elif config["dataset_name"] == "SkinCancerDataset":
        Dataset = SkinCancerDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
        ]
    )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']})")
    
    dataset  = Dataset(
        root="./data",
        split='train',
        download=True,
        transform=transform
    )

    # give the device to the deeptune based on what is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root.info(f"Device: {device}")
    root.info(f"Number of epochs: {config['epochs']}")
    report = Reporter()


    dt = DeepTune(
        seed=42,
        device=device,
        report=report,
        logger=root,
    )
    result = dt.fit(
        dataset=dataset,
        ConfigSpace=config,
        epochs=config["epochs"],
        transform=transform,
    )
    root.info("Training finished.")
    # save model checkpoint
    st_config_json_filename = config["st_config_json_filename"]
    # remove the last after last slash
    st_config_json_filename = st_config_json_filename[:st_config_json_filename.rfind("/")]
    torch.save(dt.model.state_dict(), f"{st_config_json_filename}/model.pth")
    # save the configuration
    with open(f"{st_config_json_filename}/last_run_config.json", "w") as f:
        f.write(str(config))
    root.info("Model saved.")
    report(**result)
    
