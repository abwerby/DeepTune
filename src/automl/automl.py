"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from automl.dummy_model import DinoNN, ResNet18
from automl.utils import calculate_mean_std


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(
        self,
        dataset_class: Any,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
                transforms.Resize((42, 42)), # adjust this based on the dataset
                transforms.Lambda(lambda x: x.to('cuda') if torch.cuda.is_available() else x)
            ]
        )
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        # input_size = (dataset_class.width, dataset_class.height, dataset_class.channels)
        input_size = (42, 42, 1)


        model = DinoNN(input_size, dataset_class.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")
        model.eval()
        self._model = model

        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
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
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels
