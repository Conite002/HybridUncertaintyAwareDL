import os
from torchvision import datasets

def load_image_folder(data_dir, transform=None):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    idx_to_class = { idx: cls for idx, cls in dataset.class_to_idx.items() }
    class_to_idx = { cls: idx for idx, cls in dataset.class_to_idx.items() }
    return dataset, idx_to_class, class_to_idx
