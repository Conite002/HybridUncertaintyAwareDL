import os
import random
import shutil
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import shutil
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FEATURE_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
RAW_DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD")
PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD/Preprocessed")
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32




def load_extracted_features(model_name, split):
    """
    Load extracted features from the folder
    """
    
    features = os.path.join(FEATURE_PATH, f"{model_name}_{split}_features.npy")
    labels = os.path.join(FEATURE_PATH, f"{split}_labels.npy")
    if not os.path.exists(features) or not os.path.exists(labels):
        raise FileNotFoundError(f"Features or labels not found. {features} or {labels}")
    
    X = np.load(features)
    y = np.load(labels)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_data_generator(split, transform, batch_size):
    """Loads dataset split inside each worker process."""
    dataset = ImageFolder(root=os.path.join(PREPROCESSED_PATH, split), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def create_directory_structure(base_path, with_val=False):
    splits = [ 'train', 'val' if with_val else None, 'cal', 'test' ]
    for split in splits:
        for category in os.listdir(base_path):
            os.makedirs(os.path.join(base_path.replace("Raw", "Preprocessed"), split, category), exist_ok=True)


def split_data(dataset_path, train_ratio=0.7, cal_ratio=0.1, val_ratio=0.1, test_ratio=0.1):
    """Splits dataset into train, calibration, validation, and test sets."""
    preprocessed_path = dataset_path.replace("Raw", "Preprocessed")
    
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        total = len(images)
        train_idx = int(total * train_ratio)
        cal_idx = train_idx + int(total * cal_ratio)
        val_idx = cal_idx + int(total * val_ratio)

        for i, img in enumerate(images):
            src = os.path.join(category_path, img)
            if i < train_idx:
                dst = os.path.join(preprocessed_path, "train", category, img)
            elif i < cal_idx:
                dst = os.path.join(preprocessed_path, "cal", category, img)
            elif i < val_idx:
                dst = os.path.join(preprocessed_path, "val", category, img)
            else:
                dst = os.path.join(preprocessed_path, "test", category, img)
            shutil.copy(src, dst)
            
            
        