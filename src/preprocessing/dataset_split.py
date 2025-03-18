import os
from sklearn.model_selection import train_test_split
import numpy as np




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

if __name__ == "__main__":
    import sys, os
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(PATH)
    IMAGE_FOLDER = os.path.join(PATH, "outputs/sipakmed_combine")
    SPLIT_RATIOS = {"train": 0.75, "cal": 0.1125, "val": 0.0375, "test": 0.10}
    datasets = split_dataset(IMAGE_FOLDER, SPLIT_RATIOS)
    print({key: len(value[0]) for key, value in zip(["train", "cal", "val", "test"], datasets)})
