import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision import datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import DEVICE
from data.dataset import dataloaders, data_transforms

def explain_prediction(model, image_path, method="resnet"):
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Impossible de lire l'image à : {image_path}")
        return

    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    rgb_img_normalized = np.float32(rgb_img_resized) / 255

    input_tensor = data_transforms['val'](datasets.folder.default_loader(image_path)).unsqueeze(0).to(DEVICE)

    if method == "vit":
        target_layers = [model.conv_proj]
    else:
        target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.imshow(visualization)
    plt.title(f"Analyse des artéfacts : {image_path.split('/')[-1]}")
    plt.axis('off')
    plt.show()

def test_and_explain(model, image_path, true_label, method="resnet"):
    model.eval()
    img_loader = datasets.folder.default_loader(image_path)
    input_tensor = data_transforms['val'](img_loader).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    predicted_class = dataloaders['val'].dataset.classes[preds.item()]
    is_correct = (predicted_class.lower() == true_label.lower())

    print(f"--- ANALYSE DE L'IMAGE : {image_path.split('/')[-1]} ---")
    print(f"Classe Réelle   : {true_label}")
    print(f"Classe Prédite  : {predicted_class}")
    print(f"Confiance       : {confidence.item()*100:.2f}%")
    print(f"Verdict         : {'✅ CORRECT' if is_correct else '❌ ERREUR'}")
    print("-" * 40)

    explain_prediction(model, image_path, method=method)
