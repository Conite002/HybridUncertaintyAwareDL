import torch
import torchvision.models as models
from tqdm import tqdm
import os, sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np



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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(dataloader, model):
    """Extract features from a pre-trained model."""
    features, labels_list = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu().numpy())
            labels_list.extend(labels.numpy())

    return np.vstack(features), np.array(labels_list)





if __name__ == "__main__":
    resnet = models.resnet50(pretrained=True)
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(PATH)
    from src.preprocessing.dataset_split import split_dataset
    
    SIPAKMED_COMBINE = os.path.join(PATH, "outputs/sipakmed_combine")
    SAVE_PATH = os.path.join(PATH, "outputs/feature_extraction")
    
    resnet.fc = torch.nn.Identity()
    (train_paths, train_labels), (cal_paths, cal_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        SIPAKMED_COMBINE, {
            "train": 0.75,
            "cal": 0.1125,
            "val": 0.0375,
            "test": 0.10
        }
    )
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    dataset_train = ImageDataset(train_paths, train_labels)
    dataset_val = ImageDataset(val_paths, val_labels)
    dataset_cal = ImageDataset(cal_paths, cal_labels)
    dataset_test = ImageDataset(test_paths, test_labels)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=5)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=5)
    dataloader_cal = DataLoader(dataset_cal, batch_size=32, shuffle=False, num_workers=5)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=5)

    features_train, labels_train = extract_features(dataloader_train, resnet)
    features_val, labels_val = extract_features(dataloader_val, resnet)
    features_cal, labels_cal = extract_features(dataloader_cal, resnet)
    features_test, labels_test = extract_features(dataloader_test, resnet)

    np.savez(f"{SAVE_PATH}/sipakmed_features.npz",
             train=features_train, train_labels=labels_train,
             val=features_val, val_labels=labels_val,
             cal=features_cal, cal_labels=labels_cal,
             test=features_test, test_labels=labels_test)

    print("✅ Feature extraction completed and saved.")