import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import shutil
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FEATURE_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")
RAW_DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD")
PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD/Preprocessed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def clear_directory(directory):
    """Deletes all files in a given directory while preserving the structure."""
    if os.path.exists(directory):
        for class_name in os.listdir(directory):
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    file_path = os.path.join(class_path, file)
                    os.remove(file_path)


def split_data(train_ratio=0.7, cal_ratio=0.1, val_ratio=0.1, test_ratio=0.1):
    """
    Splits dataset into train, cal, val, and test sets correctly, ensuring 
    proper distribution for each class.
    """
    
    for split in ["train", "cal", "val", "test"]:
        clear_directory(os.path.join(PREPROCESSED_PATH, split))

    total_images = 0

    for class_name in os.listdir(RAW_DATASET_PATH):
        class_dir = os.path.join(RAW_DATASET_PATH, class_name, "CROPPED")
        if not os.path.isdir(class_dir):
            continue

        images = [img for img in os.listdir(class_dir) if img.endswith(".bmp")]
        random.shuffle(images)
        
        class_total = len(images)
        total_images += class_total

        print(f"Class: {class_name} → Total images: {class_total}")

        if class_total == 0:
            print(f"Warning: No images found for class {class_name}. Skipping.")
            continue

        train_idx = int(class_total * train_ratio)
        cal_idx = train_idx + int(class_total * cal_ratio)
        val_idx = cal_idx + int(class_total * val_ratio)

        for i, img in enumerate(images):
            src = os.path.join(class_dir, img)
            if i < train_idx:
                dst = os.path.join(PREPROCESSED_PATH, "train", class_name, img)
            elif i < cal_idx:
                dst = os.path.join(PREPROCESSED_PATH, "cal", class_name, img)
            elif i < val_idx:
                dst = os.path.join(PREPROCESSED_PATH, "val", class_name, img)
            else:
                dst = os.path.join(PREPROCESSED_PATH, "test", class_name, img)
            shutil.copy(src, dst)

    print("Dataset split into train, cal, val, and test sets successfully.")
    print(f"Total images processed: {total_images}")


if __name__ == "__main__":

    splits = ["train", "cal", "val", "test"]
    for split in splits:
        for class_name in os.listdir(RAW_DATASET_PATH):
            class_dir = os.path.join(RAW_DATASET_PATH, class_name, "CROPPED")
            if os.path.isdir(class_dir):
                os.makedirs(os.path.join(PREPROCESSED_PATH, split, class_name), exist_ok=True)
                
    split_data()
    print("Data split successfully into train, cal, val, and test sets.")
    