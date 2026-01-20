import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, NUM_CLASSES

def get_resnet(pretrained=True):
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(DEVICE)
