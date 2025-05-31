import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models in the framework.
    """

    def __init__(self, model=None, device='cuda', models=None):
        self.model = model
        self.device = device
        self.models = models if models is not None else []
    @abstractmethod
    def train(self, train_loader, optimizer, criterion, epochs=1):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def get_label_preds_probs(self):
        pass