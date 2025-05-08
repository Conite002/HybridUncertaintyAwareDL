from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

    

def get_dataloaders(dataset, batch_size=32, val_split=0.03, cal_split=0.12, test_split=0.1, seed=42):
    torch.manual_seed(seed)
    
    if 'train' not in dataset:
        raise ValueError("Dataset dict must contain at least the key 'train'.")

    train_data = dataset["train"]

    dataset_size = len(train_data)
    val_size = int(dataset_size * val_split)
    cal_size = int(dataset_size * cal_split)

    # Cas 1: Seulement dossier "train" fourni, on génère "test" aussi.
    if 'test' not in dataset:
        test_size = int(dataset_size * test_split)
        train_size = dataset_size - val_size - cal_size - test_size
        train_dataset, val_dataset, cal_dataset, test_dataset = random_split(
            train_data, 
            [train_size, val_size, cal_size, test_size], 
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        # Cas 2: "test" déjà fourni, donc on ne génère pas de test à partir du train.
        train_size = dataset_size - val_size - cal_size
        train_dataset, val_dataset, cal_dataset = random_split(
            train_data, 
            [train_size, val_size, cal_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        test_dataset = dataset["test"]

    # DataLoaders correspondants
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Cal size: {len(cal_dataset)}, Test size: {len(test_dataset)}")

    return train_loader, val_loader, cal_loader, test_loader



def split_dataset(image_folder, split_ratios, random_state=42):
    """
    Splits dataset into train, calibration, validation, and test sets.
    Args:
        image_folder (str): Path to the processed images.
        split_ratios (dict): Dictionary containing split ratios.
        random_state (int): Seed for reproducibility.
    """
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.bmp')]
    labels = [int(img.split('__')[0]) for img in os.listdir(image_folder) if img.endswith('.bmp')]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=1 - split_ratios["train"], stratify=labels, random_state=random_state
    )

    remaining_ratio = 1 - split_ratios["train"]
    cal_ratio = split_ratios["cal"] / remaining_ratio
    val_ratio = split_ratios["val"] / remaining_ratio
    test_ratio = split_ratios["test"] / remaining_ratio

    cal_paths, temp_paths, cal_labels, temp_labels = train_test_split(
        temp_paths, temp_labels, test_size=(val_ratio + test_ratio), stratify=temp_labels, random_state=random_state
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_labels, random_state=random_state
    )

    return (train_paths, train_labels), (cal_paths, cal_labels), (val_paths, val_labels), (test_paths, test_labels)







PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SIPAKMED_COMBINE = os.path.join(PROJECT_ROOT, "experiments/results/sipakmed/data/sipakmed_combine")
SAVE_PATH = os.path.join(PROJECT_ROOT, "experiments/results/sipakmed/data/feature_extraction")

def get_dataloaders_for_sipakmed(
    batch_size=32, 
    num_workers=5, 
    model_type="resnet", 
    percent_split=
    {
        "train": 0.70,
        "cal": 0.12,
        "val": 0.03,
        "test": 0.15
    }):
    """Get DataLoader for training, validation, calibration, and test sets."""
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


default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image paths and labels.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        
        self.transform = transform if transform else default_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  

        return image, label
    

import pandas as pd

def extract_paths_and_labels(dataloader, subset_name):
    paths = []
    labels = []
    dataset = dataloader.dataset

    for i in range(len(dataset)):
        img_path, label = dataset.samples[i]
        paths.append(img_path)
        labels.append(label)

    df = pd.DataFrame({
        'subset': subset_name,
        'image_path': paths,
        'label': labels
    })
    return df


def dataloader_to_dataframe(dataloader, subset_name):
    dataset = dataloader.dataset
    paths = dataset.image_paths
    labels = dataset.labels

    df = pd.DataFrame({
        'subset': [subset_name] * len(paths),
        'image_path': paths,
        'label': labels
    })
    return df



def getdataframe_for_sipakmed(train_loader, val_loader, cal_loader, test_loader):
    """
    Create a dataframe for the SipakMed dataset from the dataloaders.
    """
    df_train = dataloader_to_dataframe(train_loader, "train")
    df_val   = dataloader_to_dataframe(val_loader, "val")
    df_cal   = dataloader_to_dataframe(cal_loader, "cal")
    df_test  = dataloader_to_dataframe(test_loader, "test")
    return pd.concat([df_train, df_val, df_cal, df_test], ignore_index=True)