from base_model import BaseModel
from resnet_model import initialize_resnet
import torch, numpy as np
from tqdm import tqdm
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_metrics

class MCDropout(BaseModel):
    
    def __init__(self, num_classes, dropout_rate=0.5, device='cuda'):
        model = initialize_resnet(num_classes, dropout_rate=dropout_rate, mc_dropout=True, device=device)
        super().__init__(model, device=device)
        self.dropout_rate = dropout_rate
        
        
    def train(self, train_loader, val_loader, epochs=50, lr=1e-4, weight_decay=1e-5, model_path=None):
        return self.trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            model_path=model_path
        )
    def predict(self, loader, n_samples=10):
        all_probs, all_preds, all_labels = [], [], []
        all_variances = []
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc='Predictions, leave=False'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                mc_dropout = [self.model(inputs) for _ in range(n_samples)]
                mc_dropout = torch.stack(mc_dropout, dim=0)
                probs = torch.softmax(mc_dropout, dim=2)
                mean_probs = probs.mean(dim=0)
                var_probs = probs.var(dim=0).sum(dim=1)
                
                all_probs.append(mean_probs.cpu().numpy())
                all_preds.append(torch.argmax(mean_probs, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_variances.append(var_probs.cpu().numpy())
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_variances = np.concatenate(all_variances)
        return all_probs, all_preds, all_labels, all_variances
    
    def evaluate(self, loader):
        probs, preds, labels, variances = self.predict(loader)
        metrics = compute_metrics(probs, preds, labels)
        
        return metrics
                
                    
                    
                