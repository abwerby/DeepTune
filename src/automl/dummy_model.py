from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoModel, AutoTokenizer
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os
import sys
from PIL import Image


class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.resnet18.load_state_dict(torch.load(ResNet18_Weights))
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x)


class DinoNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DinoNN, self).__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # freeze the model dinov2_vitb14
        for param in self.dino.parameters():
            param.requires_grad = False
        hidden_size = 768
        in_feat = hidden_size * input_size[0]**2 // 14**2
        self.fc = torch.nn.Linear(in_feat, num_classes)
        
    def forward(self, img):
        """
        :brief: Forward pass of the model
        :param img: image as PIL Image
        :return: output of the model
        """
        # if image is one channel, add two more channels to make it 3 channel
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        # out = self.dino.forward_features(img)
        # cls_f = out['x_norm_clstoken']
        # cls_f = out['x_norm_patchtokens']
        inter = self.dino.get_intermediate_layers(img, 5)
        cls_f = inter[0]
        cls_f = cls_f.view(cls_f.size(0), -1)
        # print(cls_f.shape)
        # print(patch_f.shape)
        # concatenate the cls token and patch tokens
        # cls_f = cls_f.unsqueeze(1)
        # x = torch.cat((cls_f, patch_f), dim=1)
        x = self.fc(cls_f)
        return x
