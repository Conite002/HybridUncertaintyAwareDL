import os
import numpy as np
import random
import json
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
import sys
sys.path.append(PROJECT_ROOT)
LOGS_PATH = os.path.join(PROJECT_ROOT, "outputs/logs")
SIPAKMED_COMBINE = os.path.join(PROJECT_ROOT, "outputs/sipakmed_combine")
SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/feature_extraction")
from src.preprocessing.dataset_split import split_dataset
from src.preprocessing.feature_extraction import ImageDataset




class MultiLabelDataset(Dataset):
    """Custom Dataset for Multi-Label Classification."""
    def __init__(self, image_paths, labels_df, transform=None):
        self.image_paths = image_paths
        self.labels_df = labels_df  # DataFrame where each column is a label
        self.transform = transform if transform else self.default_transforms()

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((299, 299)),  # Resize for InceptionV3 & EfficientNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        labels = torch.tensor(self.labels_df.iloc[idx].values, dtype=torch.float32)  # Multi-label
        if self.transform:
            image = self.transform(image)
        return image, labels



def setup_logger(model_name):
    log_file = os.path.join(LOGS_PATH, f"{model_name}_training_log.log")
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # # File handler (logs everything)
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)

    # # Suppress console logs completely
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.ERROR)

    # # Formatting
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # # Clear existing handlers if logger is re-used
    # if logger.hasHandlers():
    #     logger.handlers.clear()
    
    # # Add handlers
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_history(history, path):
    """Save training history to JSON."""
    with open(path, "w") as f:
        json.dump(history, f)




def load_history(history_path):
    """Loads training history from JSON and returns a DataFrame."""
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return None
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    return pd.DataFrame.from_dict(history)

def load_features(feature_path, split, typeofdata='sipakmed_features'):
    """Load extracted features and labels from a single .npz file."""
    data = np.load(f"{feature_path}/{typeofdata}.npz")  
    features = data[f"{split}"]
    labels = data[f"{split}_labels"]
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def get_device():
    """Get computing device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """
    Enables Dropout layers at inference time for Monte Carlo Dropout.
    This ensures that Dropout is applied during test-time sampling.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


import torchvision.transforms as transforms

IMG_SIZES = {
    "InceptionV3": 299,
    "EfficientNetB7": 600,
    "ResNet50": 224,
}
transform_inception = transforms.Compose([
    transforms.Resize((IMG_SIZES["InceptionV3"], IMG_SIZES["InceptionV3"])),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_efficientnet = transforms.Compose([
    transforms.Resize((IMG_SIZES["EfficientNetB7"], IMG_SIZES["EfficientNetB7"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_resnet = transforms.Compose([
    transforms.Resize((IMG_SIZES["ResNet50"], IMG_SIZES["ResNet50"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_dataloaders(
    batch_size=32, 
    num_workers=5, 
    model_type="inception", 
    percent_split=
    {
        "train": 0.70,
        "cal": 0.12,
        "val": 0.03,
        "test": 0.15
    }):
    """Get DataLoader for training, validation, calibration, and test sets."""
    transform = transform_inception if model_type == "inception" else transform_efficientnet if model_type == "efficientnet" else transform_resnet

    (train_paths, train_labels), (cal_paths, cal_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        SIPAKMED_COMBINE, percent_split
    )
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    dataset_train = ImageDataset(train_paths, train_labels, transform=transform)
    dataset_val = ImageDataset(val_paths, val_labels, transform=transform)
    dataset_cal = ImageDataset(cal_paths, cal_labels, transform=transform)
    dataset_test = ImageDataset(test_paths, test_labels, transform=transform)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_cal = DataLoader(dataset_cal, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_cal, dataloader_test


