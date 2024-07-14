"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple
import os
import sys

import time
import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment, TrivialAugmentWide
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from search_space.search_space import SearchSpaceParser


logger = logging.getLogger(__name__)


class SmartTune:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(
        self,
        dataset_class: Any,
        ConfigSpace: dict,
        epochs: int,
    ) -> Tuple[float, float]:
        """
            fit model with it config space to dataset for num of epoches and calculate the vaildation score and cost
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # define transformation based on config space
        # set transformation based on config space
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((42, 42)), # adjust this based on the dataset
                transforms.Lambda(lambda x: x.to(self.device))
            ]
        )
        self._transform = self._set_augmentation_hyperparameters(ConfigSpace, self._transform)
        
        # get the dataset
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=ConfigSpace["hyperparameters"]["batch_size"], shuffle=True)
        
        # set the model based on config space
        model = self._set_base_model(ConfigSpace)
        
        # set the optimizer based on config space
        optimizer = self._set_optimizer_hyperparameters(model, ConfigSpace)
        
        # TODO: train the model based on config space for num of epoches
        trainning_loss = 0
        start_time = time.time()
        for epoch in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                trainning_loss += loss.item()
            logger.info(f"Epoch {epoch} loss: {trainning_loss/len(train_loader)}")
        end_time = time.time()
        
        # get validation score
        score = self.predicit(model, dataset_class, ConfigSpace)
        logger.info(f"Validation score: {score}")
        
        # get cost as the time taken to train the model
        cost = end_time - start_time
        logger.info(f"Training time: {end_time - start_time}")

        return score, cost


    def predict(self, dataset_class, model: nn.Module) -> float:
        """
            predict the test set and return the predictions and the labels
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        score = accuracy_score(labels, predictions)
        return score        

    def _set_base_model(self, config: dict) -> nn.Module:
        """
            get the model based on the config space
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
    
    def _set_optimizer_hyperparameters(self, model: nn.Module, config: dict) -> optim.Optimizer:
        """
            get the optimizer based on the config space and set the learning rate
        """
        if config["hyperparameters"]["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(),
                                      lr=config["hyperparameters"]["learning_rate"],
                                      betas=(config["hyperparameters"]["betas"][0], config["hyperparameters"]["betas"][1]),
                                      weight_decay=config["hyperparameters"]["weight_decay"])
        elif config["hyperparameters"]["optimizer"] == "AdamW":
            optimizer = optim.AdamW(model.parameters(),
                                      lr=config["hyperparameters"]["learning_rate"],
                                        weight_decay=config["hyperparameters"]["weight_decay"],
                                        betas=(config["hyperparameters"]["betas"][0], config["hyperparameters"]["betas"][1]))
        elif config["hyperparameters"]["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["hyperparameters"]["learning_rate"],
                                  momentum=config["hyperparameters"]["momentum"],
                                  weight_decay=config["hyperparameters"]["weight_decay"])
        else:
            logger.error(f"Optimizer {config['hyperparameters']['optimizer']} not found")
            sys.exit(1)
        
        return optimizer
    
    def _set_augmentation_hyperparameters(self, config: dict, transform: transforms.Compose) -> transforms.Compose:
        """
            get the augmentation based on the config space
        """
        if config["hyperparameters"]["augmentation"] == "AutoAugment":
            transform.transforms.insert(0, AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
        elif config["hyperparameters"]["augmentation"] == "RandAugment":
            transform.transforms.insert(0, RandAugment(1, 10))
        elif config["hyperparameters"]["augmentation"] == "TrivialAugmentWide":
            transform.transforms.insert(0, TrivialAugmentWide())
        else:
            logger.error(f"Augmentation {config['hyperparameters']['augmentation']} not found")
            sys.exit(1)
        return transform
        




if __name__ == "__main__":
    model_file = "search_space/models_ss.json"
    hyperparameter_file = "search_space/hyperparameters_ss.json"
    search_space_parser = SearchSpaceParser(model_file, hyperparameter_file)
    model, hyperparameters = search_space_parser.sample_search_space()
    config = {
        "model": model,
        "hyperparameters": hyperparameters
    }
    print(config)
    tune = SmartTune(seed=42)
    tune._set_base_model(config)
