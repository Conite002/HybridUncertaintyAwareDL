import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models in the framework.
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
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