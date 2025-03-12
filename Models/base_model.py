import torch
import torch.nn as nn
import timm
from typing import Tuple, Dict, Optional
import numpy as np

class UncertaintyAwareModel(nn.Module):
    def __init__(self, 
                 model_name: str = 'resnet50',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(model_name, 
                                        pretrained=pretrained,
                                        num_classes=0)  # Remove classifier
        
        feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits, None
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 30) -> Dict[str, np.ndarray]:
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits, _ = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)
        
        # Calculate mean prediction and uncertainty metrics
        mean_probs = np.mean(predictions, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        # Calculate aleatoric and epistemic uncertainty
        expected_entropy = -np.mean(np.sum(predictions * np.log(predictions + 1e-10), axis=2), axis=0)
        aleatoric = expected_entropy
        epistemic = predictive_entropy - expected_entropy
        
        return {
            'mean_probabilities': mean_probs,
            'predictive_entropy': predictive_entropy,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic
        }
    
    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor, max_iters: int = 50) -> float:
        """Temperature scaling for model calibration"""
        temperature = nn.Parameter(torch.ones(1).cuda())
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iters)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        return temperature.item() 