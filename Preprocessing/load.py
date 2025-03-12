import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
