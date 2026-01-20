import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from config import DEVICE, NUM_CLASSES

def get_vit(pretrained=True):
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(DEVICE)
