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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)




def prepare_dataset():
    if os.path.exists(PREPROCESSED_PATH):
        print("[INFO] Dataset already split. Skipping...")
        return

    os.makedirs(PREPROCESSED_PATH, exist_ok=True)
    for split in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(PREPROCESSED_PATH, split), exist_ok=True)

    image_list = []

    for class_folder in os.listdir(CROPPED_PATH):
        class_path = os.path.join(CROPPED_PATH, class_folder, "CROPPED")

        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(".bmp")]
            image_list.extend([(os.path.join(class_path, img), class_folder) for img in images])

    random.shuffle(image_list)
    total_images = len(image_list)
    print(f"[INFO] Total images found: {total_images}")

    # Compute split sizes
    split_counts = {split: int(total_images * ratio) for split, ratio in SPLIT_RATIOS.items()}
    split_counts["test"] += total_images - sum(split_counts.values())

    # Create labels mapping
    label_records = []

    # Distribute images into splits
    start = 0
    for split, count in split_counts.items():
        print(f"[INFO] Assigning {count} images to {split}...")
        for img_path, class_name in image_list[start:start + count]:
            dest_folder = os.path.join(PREPROCESSED_PATH, split)
            os.makedirs(dest_folder, exist_ok=True) 
            
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(dest_folder, img_name))
            
            label_records.append({"image": img_name, "class": class_name, "split": split})

        start += count

    df_labels = pd.DataFrame(label_records)
    df_labels.to_csv(os.path.join(PREPROCESSED_PATH, "labels.csv"), index=False)

    print("[INFO] Dataset successfully split into train/cal/val/test!")


# 📌 Step 2: Define Custom Dataset (Uses CSV File for Labels)
class SIPaKMeDDataset(Dataset):
    def __init__(self, split, transform):
        self.data_path = os.path.join(PREPROCESSED_PATH, split)
        self.labels = pd.read_csv(os.path.join(PREPROCESSED_PATH, "labels.csv"))
        self.labels = self.labels[self.labels["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(self.labels["class"].unique()))}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.loc[idx, "image"]
        class_name = self.labels.loc[idx, "class"]
        label = self.class_to_idx[class_name]

        img_path = os.path.join(self.data_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_dataloader(split, batch_size):
    transform = train_transforms if split == "train" else val_test_transforms
    dataset = SIPaKMeDDataset(split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=6)




def get_extractor(finetune=True):
 
    model = models.resnet50(pretrained=True)
    if finetune:
 
        for layer in list(model.children()):
            for param in layer.parameters():
                param.requires_grad = True  

    else:
        for param in model.parameters():
            param.requires_grad = False
    model = nn.Sequential(*list(model.children())[:-1])  

    if finetune:
        model.train()  
    else:
        model.eval() 

    print(f"[INFO] Feature extractor loaded: ResNet50 (Fine-tune={finetune}")
    return model

def features_exist(split):
    feature_file = os.path.join(FEATURE_PATH, f"ResNet50_{split}.npy")
    return os.path.exists(feature_file)


def extract_features(extractor, split, dataloader):
    if features_exist(split):
        print(f"[INFO] Features for {split} already exist. Skipping...")
        return

    features = []
    labels = []
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc=f"Extracting {split} features"):
            features.append(extractor(images).squeeze().cpu().numpy())
            labels.append(target.cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)
    np.save(os.path.join(FEATURE_PATH, f"ResNet50_{split}.npy"), features)
    np.save(os.path.join(FEATURE_PATH, f"labels_{split}.npy"), labels)
    print(f"[INFO] Features extracted and saved for {split}.")


if __name__ == "__main__":
    # Step 1: Prepare Dataset
    prepare_dataset()

    # Step 2: Create DataLoaders
    dataloaders = {split: create_dataloader(split, BATCH_SIZE) for split in ["train", "cal", "val", "test"]}

    # Step 3: Initialize Feature Extractor
    if not os.path.exists(FEATURE_PATH):
        os.makedirs(FEATURE_PATH)
    
    extractor = get_extractor(finetune=True)

    for split, dataloader in dataloaders.items():
        extract_features(extractor, split, dataloader)
