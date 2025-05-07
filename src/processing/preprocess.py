from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

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
