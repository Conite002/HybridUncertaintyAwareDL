import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import shutil
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FEATURE_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")
RAW_DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD")
PREPROCESSED_PATH = os.path.join(PROJECT_ROOT, "Datasets/SIPaKMeD/Preprocessed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

