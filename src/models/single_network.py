from .base_model import BaseModel
from .resnet_model import initialize_resnet
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_metrics
from tqdm import tqdm
import torch
import numpy as np

class SingleNetwork(BaseModel):
    
    def __init__(self, num_classes=1, learning_rate=1e-5, patience=5, device='cuda', train_loader=None, cal_loader=None, val_loader=None, test_loader=None):
        model = initialize_resnet(num_classes, device=device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        super().__init__(model, device=device)
        self.trainer = ModelTrainer(model, device=device, learning_rate=learning_rate, patience=patience)
        
    def train(self, epochs=50, weight_decay=1e-5, model_path=None):
        return self.trainer.train(
            self.train_loader,
            self.val_loader,
            epochs=epochs,
            model_path=model_path
        )
        
    def predict(self, loader):

        all_probs, all_preds, all_labels = [], [], []
        all_variances = []
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Predicting", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_variances.append(torch.var(outputs, dim=0).cpu().numpy())
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)   
        all_labels = np.concatenate(all_labels)
        all_variances = np.concatenate(all_variances)
        return all_probs, all_preds, all_labels, all_variances
    
    def evaluate(self):
        probs, preds, labels, variances = self.predict(self.test_loader)
        metrics = compute_metrics(labels, preds, probs, variances)
        
        return metrics