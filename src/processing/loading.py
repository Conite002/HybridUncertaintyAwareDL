import os
from torchvision import datasets

def load_image_folder(data_dir, transform=None):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    idx_to_class = { idx: cls for idx, cls in dataset.class_to_idx.items() }
    class_to_idx = { cls: idx for idx, cls in dataset.class_to_idx.items() }
    return dataset, idx_to_class, class_to_idx

from torchvision import transforms
from PIL import Image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)
