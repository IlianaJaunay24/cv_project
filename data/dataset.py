from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from config import DATA_DIR, BATCH_SIZE

# Définition des transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Vérification de la structure des dossiers
train_path = os.path.join(DATA_DIR, 'train')
valid_path = os.path.join(DATA_DIR, 'valid')

if not (os.path.exists(train_path) and os.path.exists(valid_path)):
    raise FileNotFoundError("❌ Erreur : Vérifiez que 'train' et 'valid' sont bien dans DATA_DIR.")

# Chargement des datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), data_transforms['val'])
}

# Création des dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

class_names = image_datasets['train'].classes
print(f"✅ Structure confirmée. Classes détectées : {class_names}")