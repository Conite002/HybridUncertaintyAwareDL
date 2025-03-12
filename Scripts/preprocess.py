import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[A.Compose] = None,
                 phase: str = 'train'):
        """
        Dataset class for medical images
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Albumentations transformations
            phase: Dataset phase ('train', 'val', or 'test')
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.phase = phase
        
        # Basic normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = self.normalize(image)
        
        return image, self.labels[idx]

class DataPreprocessor:
    def __init__(self, config: Dict):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary containing dataset paths and parameters
        """
        self.config = config
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup image transformations for each phase"""
        # Training transforms with augmentation
        self.train_transform = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=5)
            ], p=0.1),
            A.Normalize()
        ])
        
        # Validation/Test transforms
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize()
        ])
        
    def load_dataset(self, dataset_name: str) -> Tuple[List[str], List[int]]:
        """
        Load dataset from specified path
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of image paths and labels
        """
        dataset_path = self.config['dataset_paths'][dataset_name]
        image_paths = []
        labels = []
        
        # Load images and labels based on dataset structure
        if dataset_name == 'cytology':
            # Load cervical cytology dataset
            for class_name in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_path):
                    class_label = self.config['class_mapping'][dataset_name][class_name]
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(class_path, img_name))
                            labels.append(class_label)
                            
        elif dataset_name == 'histopathology':
            # Load ovarian cancer histopathology dataset
            data_df = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))
            for _, row in data_df.iterrows():
                image_path = os.path.join(dataset_path, 'images', row['image_id'])
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(self.config['class_mapping'][dataset_name][row['label']])
                    
        elif dataset_name == 'mammography':
            # Load breast cancer mammography dataset
            for class_name in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_path):
                    class_label = self.config['class_mapping'][dataset_name][class_name]
                    for img_name in os.listdir(class_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                            image_paths.append(os.path.join(class_path, img_name))
                            labels.append(class_label)
                            
        return image_paths, labels
    
    def create_data_loaders(self,
                           dataset_name: str,
                           batch_size: int = 32,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load dataset
        image_paths, labels = self.load_dataset(dataset_name)
        
        # Split dataset
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.3, stratify=labels, random_state=42
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        # Create datasets
        train_dataset = MedicalImageDataset(
            train_paths, train_labels, transform=self.train_transform, phase='train'
        )
        val_dataset = MedicalImageDataset(
            val_paths, val_labels, transform=self.val_transform, phase='val'
        )
        test_dataset = MedicalImageDataset(
            test_paths, test_labels, transform=self.val_transform, phase='test'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example configuration
    config = {
        'dataset_paths': {
            'cytology': 'Datasets/Cytology_Cervical_Cancer',
            'histopathology': 'Datasets/Histopathology_Ovarian_Cancer',
            'mammography': 'Datasets/Mammography_Breast_Cancer'
        },
        'class_mapping': {
            'cytology': {
                'normal': 0,
                'abnormal': 1
            },
            'histopathology': {
                'benign': 0,
                'malignant': 1
            },
            'mammography': {
                'normal': 0,
                'tumor': 1
            }
        }
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Create data loaders for each dataset
    datasets = ['cytology', 'histopathology', 'mammography']
    for dataset_name in datasets:
        print(f'\nProcessing {dataset_name} dataset...')
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(dataset_name)
        print(f'Created data loaders with sizes:')
        print(f'Train: {len(train_loader.dataset)} samples')
        print(f'Validation: {len(val_loader.dataset)} samples')
        print(f'Test: {len(test_loader.dataset)} samples') 