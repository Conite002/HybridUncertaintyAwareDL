import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_metrics
from .base_model import BaseModel
from .resnet_model import initialize_resnet


class DeepEnsemble(BaseModel):

    def __init__(self, num_classes, ensemble_size=5, learning_rate=1e-5, patience=5, device='cuda'):
        self.ensemble_size = ensemble_size
        self.device = device
        self.models = [
            initialize_resnet(num_classes, device=device) 
            for _ in range(ensemble_size)
        ]
        self.trainers = [
            ModelTrainer(
                model=self.models[i], 
                device=device, 
                learning_rate=learning_rate, 
                patience=patience
            )
            for i in range(ensemble_size)
        ]

    def train(self, train_loader, val_loader, epochs=50, model_dir='ensemble_models'):
        histories = []

        for idx, trainer in enumerate(self.trainers):
            print(f"\n🚀 Training ensemble member {idx+1}/{self.ensemble_size}")
            model_path = f"{model_dir}/ensemble_model_{idx}.pth"
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                model_path=model_path
            )
            histories.append(history)
        
        return histories

    def predict(self, loader):
        all_probs = []
        all_preds = []
        all_labels = []

        for idx, model in enumerate(self.models):
            print(f"\n🔎 Predicting with ensemble member {idx+1}/{self.ensemble_size}")
            model.eval()
            model.to(self.device)

            member_probs = []
            member_preds = []

            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"Predict member {idx+1}", leave=False):
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    member_probs.append(probs.cpu().numpy())
                    member_preds.append(preds.cpu().numpy())

                    if idx == 0:
                        all_labels.append(labels.cpu().numpy())

            member_probs = np.concatenate(member_probs)
            member_preds = np.concatenate(member_preds)

            all_probs.append(member_probs)
            all_preds.append(member_preds)

        all_labels = np.concatenate(all_labels)
        avg_probs = np.mean(np.array(all_probs), axis=0)
        majority_preds = np.argmax(avg_probs, axis=1)
        variance_across_members = np.var(np.array(all_probs), axis=0)

        return {
            "ensemble_probs": avg_probs,
            "ensemble_preds": majority_preds,
            "all_labels": all_labels,
            "member_preds": all_preds,
            "variance_across_members": variance_across_members
        }

    def evaluate(self, loader):
        results = self.predict(loader)
        metrics = compute_metrics(
            results["all_labels"],
            results["ensemble_preds"],
            results["ensemble_probs"]
        )
        metrics["mean_variance"] = results["variance_across_members"].mean()

        return metrics
