import os
import numpy as np
import torch
import random
import json
import logging
# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_history(history, path):
    """Save training history to JSON."""
    with open(path, "w") as f:
        json.dump(history, f)


def load_features(feature_path, split):
    """Load extracted features and labels from a single .npz file."""
    data = np.load(f"{feature_path}/sipakmed_features.npz")  
    features = data[f"{split}"]
    labels = data[f"{split}_labels"]
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def get_device():
    """Get computing device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """
    Enables Dropout layers at inference time for Monte Carlo Dropout.
    This ensures that Dropout is applied during test-time sampling.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
