import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import shutil


DATASET_PATH = os.path.join(os.path.dirname(__file__), "../Datasets/SIPaKMeD")

IMG_SIZE = (224, 224)  
BATCH_SIZE = 32


def create_directory_structure(base_path, with_val=False):
    splits = [ 'train', 'val' if with_val else None, 'cal', 'test' ]
    for split in splits:
        for category in os.listdir(base_path):
            os.makedirs(os.path.join(base_path.replace("Raw", "Preprocessed"), split, category), exist_ok=True)


def split_data(dataset_path, train_ratio=0.7, cal_ratio=0.1, val_ratio=0.1, test_ratio=0.1):
    """Splits dataset into train, calibration, validation, and test sets."""
    preprocessed_path = dataset_path.replace("Raw", "Preprocessed")
    
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        total = len(images)
        train_idx = int(total * train_ratio)
        cal_idx = train_idx + int(total * cal_ratio)
        val_idx = cal_idx + int(total * val_ratio)

        for i, img in enumerate(images):
            src = os.path.join(category_path, img)
            if i < train_idx:
                dst = os.path.join(preprocessed_path, "train", category, img)
            elif i < cal_idx:
                dst = os.path.join(preprocessed_path, "cal", category, img)
            elif i < val_idx:
                dst = os.path.join(preprocessed_path, "val", category, img)
            else:
                dst = os.path.join(preprocessed_path, "test", category, img)
            shutil.copy(src, dst)
            
            
            

def load_data(dataset_path, img_size=(224, 224), batch_size=32, use_val=True):
    """Loads dataset with an option to include validation data."""
    
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen_train.flow_from_directory(
        dataset_path + "/train", 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical"
    )

    cal_data = datagen.flow_from_directory(
        dataset_path + "/cal", 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical"
    )

    val_data = None
    if use_val:
        val_data = datagen.flow_from_directory(
            dataset_path + "/val", 
            target_size=img_size, 
            batch_size=batch_size, 
            class_mode="categorical"
        )

    test_data = datagen.flow_from_directory(
        dataset_path + "/test", 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical", 
        shuffle=False
    )

    return train_data, cal_data, val_data, test_data
