from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import sys
from PIL import Image



class DummyNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DummyNN, self).__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # freeze the model dinov2_vitb14
        for param in self.dino.parameters():
            param.requires_grad = False
        hidden_size = 768
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, img):
        """
        :brief: Forward pass of the model
        :param img: image as PIL Image
        :return: output of the model
        """
        out = self.dino.forward_features(img)
        cls_f = out['x_norm_clstoken']
        patch_f = out['x_norm_patchtokens']
        # concatenate the cls token and patch tokens
        # cls_f = cls_f.unsqueeze(1)
        # x = torch.cat((cls_f, patch_f), dim=1)
        x = self.fc(cls_f)
        return x
