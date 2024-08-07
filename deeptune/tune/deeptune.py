# TODO: change the file comment
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from syne_tune import Reporter
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment, TrivialAugmentWide
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# uncomment if you want to test the deeptune class
# from tune.datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from search_space.search_space import SearchSpace
from utils.constants import MODELS_MAX_LAYERS


class DeepTune:

    def __init__(
        self,
        seed: int,
        device: str,
        report: Reporter = None,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.seed = seed
        self.device = device
        self.logger = logger
        self.report = report

    def fit(
        self,
        dataset: Any,
        ConfigSpace: dict,
        epochs: int,
        transform: transforms.Compose = None,
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
        
        # size of the dataset
        self.logger.info(f"Size of the dataset: {len(dataset)}")
        
        # create a dataloader
        batch_size = int(ConfigSpace["batch_size"])
        
        # split the dataset into train and validation
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(range(0, int(0.1*len(dataset)))))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = DataLoader(val_dataset, batch_size=32, shuffle=False)
        # val_dataset = DataLoader(val_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(range(0, int(0.1*len(dataset)))))


        in_size = train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]
        # set the model based on config space
        base_model = self._set_base_model(ConfigSpace).to(self.device)
        # self.model = self._set_side_model1(base_model, ConfigSpace, dataset.num_classes)
        self.model = self._set_side_model2(input_size=in_size, base_model=base_model, config=ConfigSpace, num_classes=dataset.num_classes)
        # self.model = self._set_linear_layer(base_model, dataset.num_classes)
        
        # set the loss function to cross entropy as it is a classification task
        self.criterion = nn.CrossEntropyLoss()
        
        # set the optimizer based on config space
        optimizer = self._set_optimizer_hyperparameters(self.model, ConfigSpace)
        # optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if ConfigSpace["opt"] != "adam" and ConfigSpace["opt"] != "adamw":
            if ConfigSpace["sched"] != "None":
                lr_scheduler = self._get_learning_rate_scheduler(optimizer, ConfigSpace)
    
        result = {}        
        # train the model
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            epoch_train_loss = []
            epoch_val_loss = []
            self.model.train()
            for i, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Step the scheduler
                if ConfigSpace["opt"] != "adam" and ConfigSpace["opt"] != "adamw":
                    if ConfigSpace["sched"] != "None" and ConfigSpace["sched"] != "plateau":
                        lr_scheduler.step()
                    elif ConfigSpace["sched"] != "None" and ConfigSpace["sched"] == "plateau":
                        lr_scheduler.step(loss)

                epoch_train_loss.append(loss.item())
            # get the validation loss
            self.model.eval()
            with torch.no_grad():
                for data, target in val_dataset:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    epoch_val_loss.append(loss.item())
            self.logger.info(f"Epoch: {epoch} Train loss: {np.mean(epoch_train_loss)} Val loss: {np.mean(epoch_val_loss)}")
            result = {
                "epoch": epoch+1,
                "train_loss": np.mean(epoch_train_loss),
                "val_loss": np.mean(epoch_val_loss),
                "time": time.time() - start_time,
            }
            if self.report is not None:
                self.report(**result)
        
        return result


    def predict(self, dataset_class, model: nn.Module, transform, save=False) -> float:
        """
            predict the test set and return the predictions and the labels
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.cpu().detach().numpy())
                predictions.append(predicted.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        # save the predictions as npy file
        if save:
            output_dir = "data/exam_dataset"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "predictions.npy"), predictions)
        labels = np.concatenate(labels)
        # check if predictions are not nans
        if np.isnan(labels).any():
            self.logger.warning("Predictions contain nans")
            return 0
        score = accuracy_score(labels, predictions)
        return score
    
    def load_model(self, model_path: str, config: dict, input_size, num_classes: int):
        """
            load the model from the given path
        """
        # create the model based on the config space
        base_model = self._set_base_model(config).to(self.device)
        # set the linear layer based on the config space
        self.model = self._set_side_model2(base_model, input_size, config, num_classes=num_classes)
        # load the model from the given path
        self.model.load_state_dict(torch.load(model_path))

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
            self.logger.error(f"Model {config['model']} not found")
            sys.exit(1)
        for i, param in enumerate(model.parameters()):
            param.requires_grad = False
        return model
    
    # TODO: make the linear layer more generic to accept any model
    def _set_linear_layer(self, model: nn.Module, num_classes: int) -> nn.Module:
        """
            add a linear layer to the model based on the number of classes
        """
        class classifier(nn.Module):
            def __init__(self, model, num_classes):
                super(classifier, self).__init__()
                self.base_model = model
                self.fc = nn.Linear(768, num_classes)
            def forward(self, x):
                x = self.base_model.forward_features(x)
                x = x['x_norm_clstoken']
                x = self.fc(x)
                return x
        return classifier(model, num_classes).to(self.device)
    
    def _set_side_model(self, base_model, config: dict, num_classes) -> nn.Module:
        """
            create a side model based on the config space, sniff features from the base model at a certain layer
            based on the config space, how many layers it sniff from the base model, and how many layers it has
            concatenate the features and pass it to a linear layer to get the output
        """
        class SideModel(nn.Module):
            def __init__(self, base_model, num_classes, config):
                super(SideModel, self).__init__()
                self.base_model = base_model
                self.num_classes = num_classes
                # self.num_intermediate_features = int(config['number_of_intermediate_features'])
                self.intermediate_layer_idx = config['intermediate_layer_idx'].split(" ")
                self.intermediate_layer_idx = [int(i) for i in self.intermediate_layer_idx]
                self.num_intermediate_features = len(self.intermediate_layer_idx)
                # Create a list of linear layers for each intermediate feature
                self.linear_layers = nn.ModuleList([
                    nn.Linear(768, 32) for _ in range(self.num_intermediate_features)
                ])
                # Final classification layer
                self.fc = nn.Linear(self.num_intermediate_features * 32, num_classes)

            def forward(self, x):
                out = self.base_model.get_intermediate_layers(x, MODELS_MAX_LAYERS['dinov2_vitb14'], return_class_token=True)
                processed_feats = []
                for i, idx in enumerate(self.intermediate_layer_idx):
                    # Feed each feature through its corresponding linear layer
                    processed_feat = self.linear_layers[i](out[idx][1])
                    processed_feats.append(processed_feat)
                # Concatenate the processed features
                concat_feats = torch.cat(processed_feats, dim=-1)
                # Pass the concatenated features to the final linear layer
                out = self.fc(concat_feats)
                return out
        return SideModel(base_model, num_classes, config).to(self.device)

    def _set_side_model2(self, base_model, input_size, config: dict, num_classes) -> nn.Module:
        """
            create a side model based on the config space, sniff features from the base model at a certain layer
            based on the config space, how many layers it sniff from the base model, and how many layers it has
            concatenate the features and pass it to a linear layer to get the output
        """
        class SideModel(nn.Module):
            def __init__(self, base_model, input_size, num_classes, config):
                super(SideModel, self).__init__()
                self.base_model = base_model
                self.num_classes = num_classes
                # self.num_intermediate_features = int(config['number_of_intermediate_features'])
                self.intermediate_layer_idx = config['intermediate_layer_idx'].split(" ")
                self.intermediate_layer_idx = [int(i) for i in self.intermediate_layer_idx]
                self.num_intermediate_features = len(self.intermediate_layer_idx)
                self.in_layer = nn.Linear(input_size, 32)
                # Create a list of linear layers for each intermediate feature
                self.linear_layers = nn.ModuleList([
                    nn.Linear(768 + 32, 32) for i in range(self.num_intermediate_features)
                ])
                # Final classification layer
                self.fc = nn.Linear(32, num_classes)

            def forward(self, x):
                base_model_out = self.base_model.get_intermediate_layers(x, MODELS_MAX_LAYERS['dinov2_vitb14'], return_class_token=True)
                x = self.in_layer(x.view(x.size(0), -1))
                # feed the intermediate features to the linear layers
                for i, idx in enumerate(self.intermediate_layer_idx):
                    x = torch.cat((x, base_model_out[idx][1]), dim=1)
                    x = self.linear_layers[i](x)
                    x = F.relu(x)
                x = self.fc(x)
                return x
        return SideModel(base_model, input_size, num_classes, config).to(self.device)
    
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
            self.logger.error(f"Optimizer {config['opt']} not found")
            sys.exit(1)
        
        return optimizer
    
    def _get_learning_rate_scheduler(self, optimizer: optim.Optimizer, config: dict) -> optim.lr_scheduler:
        """
            get the learning rate scheduler based on the config space
        """
        if config["sched"] == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"])
        elif config["sched"] == "multistep":
            milestones = config["milestones"].split(" ")
            milestones = [int(i) for i in milestones]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        elif config["sched"] == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer)
        elif config["sched"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"], eta_min=config["eta_min"])
        elif config["sched"] == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config["mode"], factor=config["factor"], patience=config["patience_epochs"])
        else:
            self.logger.error(f"Learning rate scheduler {config['sched']} not found")
            sys.exit(1)
        return scheduler

    @staticmethod
    def set_augmentation_hyperparameters(config: dict, transform: transforms.Compose) -> transforms.Compose:
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
            sys.exit(1)
        return transform
        

if __name__ == "__main__":
    
    # just for testing the deeptune class
    ss = SearchSpace("search_space/search_space_v1.yml")
    config, args = ss.sample_configuration(return_args=True)
    print(config)
    
    logger = logging.getLogger()
    tune = DeepTune(seed=42, device='cuda', logger=logger)
    # define base transformation based on the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((350, 350)),
            # repeat the image 3 times to make it 3 channels
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )
    
    # transform = DeepTune.set_augmentation_hyperparameters(config=config, transform=transform)
    
    dataset = SkinCancerDataset(
        root="./data",
        split='train',
        download=True,
        transform=transform
    )
    tune.fit(dataset=dataset, ConfigSpace=config, epochs=50)
    
    # get the test score
    score = tune.predict(SkinCancerDataset, tune.model, transform)
    print(f"Test score: {score}")