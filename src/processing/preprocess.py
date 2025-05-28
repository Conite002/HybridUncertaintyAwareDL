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



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SIPAKMED_COMBINE = os.path.join(PROJECT_ROOT, "experiments/results/sipakmed/data/sipakmed_combine")
SAVE_PATH = os.path.join(PROJECT_ROOT, "experiments/results/sipakmed/data/feature_extraction")



def get_transforms(model_type="resnet", augment=False):
    IMG_SIZES = {
        "resnet": 224,
        "inception": 299,
        "efficientnet": 600,
    }
    image_size = IMG_SIZES.get(model_type.lower(), 224)

    base_transform = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    if augment:
        augmentations = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        return transforms.Compose(augmentations + base_transform)
    
    return transforms.Compose(base_transform)

def split_dataset_from_folder(image_folder, split_ratios, random_state=42):
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


def get_dataloaders_from_paths(
    split_data, 
    model_type="resnet", 
    batch_size=32, 
    num_workers=4,
    augment=False
):
    transform = get_transforms(model_type=model_type, augment=augment)

    dataset_train = ImageDataset(*split_data[0], transform=transform)
    dataset_cal   = ImageDataset(*split_data[1], transform=transform)
    dataset_val   = ImageDataset(*split_data[2], transform=transform)
    dataset_test  = ImageDataset(*split_data[3], transform=transform)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_cal   = DataLoader(dataset_cal, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loader_val   = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader_train, loader_val, loader_cal, loader_test


def get_dataloaders_for_sipakmed(batch_size=32, num_workers=4, model_type="resnet"):
    split_ratios = {
        "train": 0.70,
        "cal": 0.12,
        "val": 0.03,
        "test": 0.15
    }

    split_data = split_dataset_from_folder(SIPAKMED_COMBINE, split_ratios)
    return get_dataloaders_from_paths(split_data, model_type=model_type, batch_size=batch_size, num_workers=num_workers)


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image paths and labels.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform if transform else default_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  

        return image, label
    

def get_dataloaders_for_ovarian(dataset_dict, batch_size=32, model_type="resnet", val_split=0.03, cal_split=0.12, seed=42):
    torch.manual_seed(seed)

    train_data = dataset_dict["train"]
    test_data = dataset_dict["test"]

    dataset_size = len(train_data)
    val_size = int(dataset_size * val_split)
    cal_size = int(dataset_size * cal_split)
    train_size = dataset_size - val_size - cal_size

    train_subset, val_subset, cal_subset = random_split(
        train_data,
        [train_size, val_size, cal_size],
        generator=torch.Generator().manual_seed(seed)
    )

    transform = get_transforms(model_type=model_type)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    cal_loader   = DataLoader(cal_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset_dict["test"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, cal_loader, test_loader




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
    # {im_Dyskeratotic: 0, im_Koilocytic: 1, im_Metaplasticimage: 2, im_Parabasal:3, im_Superficial-Intermediate: 4}
    label_mapping = {
        0: "im_Dyskeratotic",
        1: "im_Koilocytic",
        2: "im_Metaplastic",
        3: "im_Parabasal",
        4: "im_Superficial-Intermediate"
    }
    df = pd.DataFrame({
        'subset': [subset_name] * len(paths),
        'image_path': paths,
        'label_id': labels,
        'label_name': [label_mapping[label] for label in labels]
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