import logging
import json
import torch
import os
import hydra
from omegaconf import DictConfig
from typing import Optional, Dict, Any
from argparse import ArgumentParser
from torchvision import transforms

from syne_tune.utils import add_config_json_to_argparse, load_config_json


from deeptune.tune.deeptune import DeepTune
from deeptune.tune.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset


@hydra.main(config_path="config", config_name="test_config")
def main(cfg: DictConfig):
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    # read the config file
    best_trial_path = cfg.best_trial_path
    config = json.load(open(os.path.join(best_trial_path, "config.json")))
    config["st_config_json_filename"] = os.path.join(best_trial_path, "config.json")
    config = load_config_json(config)

    root.info("Starting training...")
    root.info("Config: %s", config)

    # define base transformation based on the dataset
    if cfg.dataset_name == "FashionDataset":
        Dataset = FashionDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )
    elif cfg.dataset_name == "FlowersDataset":
        Dataset = FlowersDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
        ]
    )
    elif cfg.dataset_name == "EmotionsDataset":
        Dataset = EmotionsDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
        ]
    )
    elif cfg.dataset_name == "SkinCancerDataset":
        Dataset = SkinCancerDataset
        transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
        ]
    )
    
    dataset  = Dataset(
        root="./data",
        split='train',
        download=True,
        transform=transform
    )
    
    in_size = dataset[0][0].shape[0] * dataset[0][0].shape[1] * dataset[0][0].shape[2]
    # give the device to the deeptune based on what is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root.info(f"Device: {device}")

    dt = DeepTune(
        seed=42,
        device=device,
        logger=root,
    )
    
    dt.load_model(os.path.join(best_trial_path, "model.pth"), config=config, input_size=in_size, num_classes=dataset.num_classes)
    

    # test the model
    root.info("Starting testing...")    
    # get the test score
    score = dt.predict(Dataset, dt.model, transform, save=True)
    print(f"Test score: {score}")



if __name__ == "__main__":
    main()