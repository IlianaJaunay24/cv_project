import torch

# Chemins et paramètres globaux
DATA_DIR = r'rvf10k'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS_RESNET = 10
NUM_EPOCHS_VIT = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 2
