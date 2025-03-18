import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import random
import shutil
from PIL import Image
import os
import shutil
from tqdm import tqdm

# 📌 Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FEATURE_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction/")
RAW_DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD/")
CROPPED_PATH = RAW_DATASET_PATH  # Images are inside class folders in CROPPED
PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "Outputs/SIPaKMeD_Split")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

SPLIT_RATIOS = {"train": 0.75, "cal": 0.1125, "val": 0.0375, "test": 0.10}
# SPLIT_RATIOS = {"train": 0.70, "cal": 0.10, "val": 0.10, "test": 0.10}



def organize_images(base_dir, output_dir, folders):
    """
    Organize and rename images for consistency.
    Args:
        base_dir (str): Path to the raw dataset directory.
        output_dir (str): Path to save organized dataset.
        folders (list): List of class folders.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_mapping = {folder: i for i, folder in enumerate(folders)}

    for folder in folders:
        class_dir = os.path.join(base_dir, folder, "CROPPED")
        
        if os.path.exists(class_dir):
            for file in tqdm(os.listdir(class_dir), desc=f"Processing {folder}"):
                if file.endswith(".bmp"):
                    old_path = os.path.join(class_dir, file)
                    new_filename = f"{label_mapping[folder]}__{file.split('.')[0]}.bmp"
                    new_path = os.path.join(output_dir, new_filename)
                    shutil.copy(old_path, new_path)

    print("✅ Image organization completed.")

if __name__ == "__main__":
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    BASE_DIR = os.path.join(PATH, "datasets/SIPaKMeD")
    OUTPUT_DIR = os.path.join(PATH, "outputs/sipakmed_combine")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
    organize_images(BASE_DIR, OUTPUT_DIR, CLASSES)
