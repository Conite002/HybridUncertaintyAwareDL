import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.base_model import UncertaintyAwareModel
from Conformal_Prediction.conformal_predictor import ConformalPredictor
from Explainability.score_cam import ScoreCAM
from Evaluation.metrics import UncertaintyMetrics, ClassificationMetrics, compute_spearman_correlation

class Trainer:
    def __init__(self,
                 model: UncertaintyAwareModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict):
        """
        Initialize the trainer
        
        Args:
            model: The uncertainty-aware model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=5, verbose=True
        )
        
        # Initialize metrics
        self.classification_metrics = ClassificationMetrics()
        self.uncertainty_metrics = UncertaintyMetrics()
        
        # Initialize conformal predictor
        self.conformal_predictor = ConformalPredictor(
            alpha=config['conformal_alpha'],
            method=config['conformal_method']
        )
        
        # Initialize ScoreCAM
        self.score_cam = ScoreCAM(self.model, target_layer=config['target_layer'])
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup W&B logging"""
        wandb.init(
            project=self.config['project_name'],
            name=self.config['run_name'],
            config=self.config
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.train_loader, desc='Training'):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _ = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        metrics = self.classification_metrics.compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs)
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def evaluate(self, loader: DataLoader, phase: str) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_uncertainties = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Evaluating {phase}'):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions and uncertainties
                outputs = self.model.predict_with_uncertainty(images, num_samples=30)
                probs = outputs['mean_probabilities']
                uncertainties = outputs['predictive_entropy']
                
                loss = self.criterion(torch.tensor(probs).to(self.device), labels)
                total_loss += loss.item()
                
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                all_uncertainties.extend(uncertainties)
                
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_uncertainties = np.array(all_uncertainties)
        
        # Compute metrics
        metrics = {}
        
        # Classification metrics
        metrics.update(self.classification_metrics.compute_metrics(
            all_preds, all_labels, all_probs
        ))
        
        # Calibration metrics
        metrics['ece'] = self.uncertainty_metrics.expected_calibration_error(
            all_probs, all_labels
        )
        metrics['brier_score'] = self.uncertainty_metrics.brier_score(
            all_probs, all_labels
        )
        
        # Uncertainty correlation with accuracy
        correct_mask = all_preds == all_labels
        effect_sizes = self.uncertainty_metrics.compute_effect_sizes(
            all_uncertainties[correct_mask],
            all_uncertainties[~correct_mask]
        )
        metrics.update(effect_sizes)
        
        # Conformal prediction (if validation phase)
        if phase == 'val':
            self.conformal_predictor.calibrate(all_probs, all_labels)
        elif phase == 'test':
            pred_sets, set_sizes = self.conformal_predictor.predict(all_probs)
            cp_metrics = self.conformal_predictor.evaluate(pred_sets, all_labels)
            metrics.update(cp_metrics)
            
            # Correlation between uncertainty and set size
            metrics['uncertainty_set_correlation'] = compute_spearman_correlation(
                all_uncertainties, set_sizes
            )
            
        metrics['loss'] = total_loss / len(loader)
        return metrics
    
    def train(self, num_epochs: int):
        """Train the model"""
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Training phase
            train_metrics = self.train_epoch()
            wandb.log({f'train/{k}': v for k, v in train_metrics.items()})
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader, 'val')
            wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
            
            # Update learning rate
            self.scheduler.step(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(),
                         os.path.join(self.config['output_dir'], 'best_model.pth'))
                
        # Final evaluation on test set
        self.model.load_state_dict(torch.load(
            os.path.join(self.config['output_dir'], 'best_model.pth')
        ))
        test_metrics = self.evaluate(self.test_loader, 'test')
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
        
        # Generate and save explanations
        self.generate_explanations()
        
    def generate_explanations(self, num_samples: int = 10):
        """Generate and save explanations for sample predictions"""
        self.model.eval()
        explanations_dir = os.path.join(self.config['output_dir'], 'explanations')
        os.makedirs(explanations_dir, exist_ok=True)
        
        # Get random samples from test set
        test_samples = []
        test_labels = []
        for images, labels in self.test_loader:
            test_samples.extend(images)
            test_labels.extend(labels)
            if len(test_samples) >= num_samples:
                break
                
        test_samples = test_samples[:num_samples]
        test_labels = test_labels[:num_samples]
        
        for i, (image, label) in enumerate(zip(test_samples, test_labels)):
            # Get prediction and uncertainty
            image_tensor = image.unsqueeze(0).to(self.device)
            outputs = self.model.predict_with_uncertainty(image_tensor)
            pred = np.argmax(outputs['mean_probabilities'])
            uncertainty = outputs['predictive_entropy'][0]
            
            # Generate CAM
            cam = self.score_cam.generate_cam(image_tensor, target_class=pred)
            
            # Save original image and CAM overlay
            image_np = image.permute(1, 2, 0).numpy()
            overlay = self.score_cam.overlay_cam(image_np, cam)
            
            # Save results
            np.save(os.path.join(explanations_dir, f'sample_{i}_cam.npy'), cam)
            np.save(os.path.join(explanations_dir, f'sample_{i}_overlay.npy'), overlay)
            
            # Log to W&B
            wandb.log({
                f'explanations/sample_{i}': wandb.Image(overlay),
                f'explanations/sample_{i}_uncertainty': uncertainty,
                f'explanations/sample_{i}_pred': pred,
                f'explanations/sample_{i}_true': label
            })

if __name__ == '__main__':
    # Example usage
    config = {
        'project_name': 'medical_uncertainty',
        'run_name': 'experiment_1',
        'learning_rate': 1e-4,
        'conformal_alpha': 0.1,
        'conformal_method': 'aps',
        'target_layer': 'layer4',
        'output_dir': 'Outputs/models'
    }
    
    # Initialize model and data loaders
    model = UncertaintyAwareModel(model_name='resnet50', num_classes=2)
    
    # TODO: Initialize your data loaders here
    train_loader = None
    val_loader = None
    test_loader = None
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    
    # Train model
    trainer.train(num_epochs=50)