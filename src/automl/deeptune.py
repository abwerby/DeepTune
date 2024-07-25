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
from tqdm import tqdm

import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment, TrivialAugmentWide
from sklearn.metrics import accuracy_score


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from search_space.search_space import SearchSpace


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO)

class DeepTune:

    def __init__(
        self,
        seed: int,
        device: str,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.seed = seed
        self.device = device
        self.logger = logger

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
                transforms.Resize((350, 350)), # adjust this based on the dataset
                transforms.ToTensor(),
            ]
        )
        self._transform = self._set_augmentation_hyperparameters(ConfigSpace, self._transform)
        dataset_class.transform = self._transform
        
        # size of the dataset
        self.logger.info(f"Size of the dataset: {len(dataset_class)}")
        
        # get the dataset
        # dataset = dataset_class(
        #     root="./data",
        #     split='train',
        #     download=True,
        #     transform=self._transform
        # )
        batch_size = int(ConfigSpace["batch_size"])
        train_loader = DataLoader(dataset_class, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(range(0, int(0.01*len(dataset_class)))))
        
        # set the model based on config space
        model = self._set_base_model(ConfigSpace).to(self.device)
        
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
            self.logger.info(f"Epoch {epoch} loss: {trainning_loss/len(train_loader)}")
        end_time = time.time()
        
        # get validation score
        score = 0
        # score = self.predicit(model, dataset_class, ConfigSpace)
        # logger.info(f"Validation score: {score}")
        
        # get cost as the time taken to train the model
        cost = end_time - start_time
        self.logger.info(f"Training time: {end_time - start_time}")

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
        if config["opt"] == "adam":
            betas = config["opt_betas"].split(" ")
            betas = (float(betas[0]), float(betas[1]))
            optimizer = optim.Adam(model.parameters(),
                                      lr=config["lr"],
                                      betas=betas,
                                      weight_decay=config["weight_decay"])
        elif config["opt"] == "adamw":
            betas = config["opt_betas"].split(" ")
            betas = (float(betas[0]), float(betas[1]))
            optimizer = optim.AdamW(model.parameters(),
                                      lr=config["lr"],
                                        weight_decay=config["weight_decay"],
                                        betas=betas)
        elif config["opt"] == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"])
        elif config["opt"] == "momentum":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["lr"],
                                  momentum=config["momentum"],
                                  weight_decay=config["weight_decay"])
        else:
            logger.error(f"Optimizer {config['opt']} not found")
            sys.exit(1)
        
        return optimizer
    
    def _set_augmentation_hyperparameters(self, config: dict, transform: transforms.Compose) -> transforms.Compose:
        """
            get the augmentation based on the config space
        """
        if config["data_augmentation"] == "auto_augment":
            transform.transforms.insert(0, AutoAugment())
        elif config["data_augmentation"] == "random_augment":
            transform.transforms.insert(0, RandAugment(config['ra_num_ops'], config['ra_magnitude']))
        elif config["data_augmentation"] == "trivial_augment":
            transform.transforms.insert(0, TrivialAugmentWide())
        elif config["data_augmentation"] == "None":
            pass
        else:
            logger.error(f"Data augmentation {config['data_augmentation']} not found")
            sys.exit(1)
        return transform
        

if __name__ == "__main__":
    
    ss = SearchSpace("search_space/search_space_v1.yml")
    config, args = ss.sample_configuration(return_args=True)
    print(config)
    tune = DeepTune(seed=42, device='cuda')
    dataset = torchvision.datasets.CIFAR10("data", train=True, download=True)
    tune.fit(dataset_class=dataset, ConfigSpace=config, epochs=1)