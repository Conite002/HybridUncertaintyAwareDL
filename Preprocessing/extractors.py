import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter



from tqdm import tqdm
from multiprocessing import Pool, cpu_count
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)
from Datasets.split import split_data
from Preprocessing.preprocessing import load_data_generator
OUTPUT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")


class VGG_Lite(nn.Module):
    def __init__(self):
        super(VGG_Lite, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x


def get_transforms(model_name):
    """
    Returns the appropriate image transformations for the given model.
    Ensures images are resized correctly for feature extraction.
    """
    if model_name == "InceptionV3":
        min_size = 75  
    else:
        min_size = 224  

    return transforms.Compose([
        transforms.Resize((min_size, min_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_feature_extractor(model_name):
    """
    Returns a feature extractor model with the final classification layer removed.
    """
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
        feature_extractor = nn.Sequential(
            *list(model.children())[:-2],  
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        feature_dim = model.fc.in_features  

    elif model_name == "InceptionV3":
        model = models.inception_v3(pretrained=True, aux_logits=False)
        feature_extractor = nn.Sequential(
            *list(model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        feature_dim = model.fc.in_features  

    elif model_name == "EfficientNetB7":
        model = models.efficientnet_b7(pretrained=True)
        feature_extractor = nn.Sequential(
            *list(model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        feature_dim = model.classifier[1].in_features  

    elif model_name == "VGG_Lite":
        feature_extractor = VGG_Lite()
        feature_dim = 512

    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from: ResNet50, InceptionV3, EfficientNetB7, VGG_Lite")

    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor, feature_dim


def check_class_distribution(dataloader, split_name):
    """
    Prints class distribution of dataset split.
    """
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.cpu().numpy())

    class_counts = Counter(all_labels)
    print(f"\n[DEBUG] {split_name} Class Distribution: {class_counts}")



def extract_features_task(args):
    """
    Runs feature extraction for a specific model and dataset split in parallel.
    """
    model_name, split_name, output_path = args
    print(f"\n[INFO] Starting feature extraction: {model_name} | {split_name}")

    feature_path = os.path.join(output_path, f"{model_name}_{split_name}_features.npy")
    label_path = os.path.join(output_path, f"{split_name}_labels.npy")

    if os.path.exists(feature_path) and os.path.exists(label_path):
        print(f"[INFO] Skipping {model_name} | {split_name} (Already Extracted)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, feature_dim = get_feature_extractor(model_name)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    transform = get_transforms(model_name)
    dataloader = load_data_generator(split_name, transform=transform, batch_size=32) 

    features = []
    labels = []
    check_class_distribution(dataloader, split_name)
    
    
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (batch_images, batch_labels) in enumerate(tqdm(dataloader, desc=f"{model_name} | {split_name}")):
            batch_images = batch_images.to(device)

            batch_features = feature_extractor(batch_images)
            batch_features = batch_features.view(batch_features.size(0), -1)

            if i == 0:
                print(f"[DEBUG] {model_name} | {split_name} - Input shape: {batch_images.shape}")
            
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu())

            if (i + 1) % max(1, num_batches // 20) == 0:
                print(f"[INFO] {model_name} | {split_name}: Processed {i+1}/{num_batches} batches...")

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    np.save(feature_path, features)
    np.save(label_path, labels)

    print(f"[INFO] Completed: {model_name} | {split_name} | Features: {features.shape} | Labels: {labels.shape}")


def process_feature_extraction(models_names=["ResNet50", "DenseNet121", "VGG_Lite", "EfficientNetB7", "InceptionV3"], num_workers=None):
    """
    Runs feature extraction in parallel for all models and dataset splits using multiprocessing.
    """
    os.makedirs(OUTPUT_FEATURES_PATH, exist_ok=True)
    splits = ["train", "cal", "val", "test"]
    
    if num_workers is None:
        num_workers = max(1, cpu_count()-10)

    tasks = [(model_name, split_name, OUTPUT_FEATURES_PATH) for model_name in models_names for split_name in splits]

    print(f"\n[INFO] Running feature extraction using {num_workers} parallel processes...")

    with Pool(processes=num_workers) as pool:
        pool.map(extract_features_task, tasks)

    print("\n[INFO] All feature extraction completed successfully.")


if __name__ == "__main__":
    process_feature_extraction(models_names=["ResNet50", "EfficientNetB7", 'VGG_Lite'], num_workers=4)
