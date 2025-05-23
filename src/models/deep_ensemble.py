import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_metrics
from .base_model import BaseModel
from .resnet_model import initialize_resnet


class DeepEnsemble(BaseModel):

    def __init__(self, num_classes, ensemble_size=5, learning_rate=1e-5, patience=5, device='cuda', train_loader=None, cal_loader=None, val_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        
        self.ensemble_size = ensemble_size
        self.device = device
        self.models = [
            initialize_resnet(num_classes, device=device) 
            for _ in range(ensemble_size)
        ]
        super().__init__(models=self.models, device=device)
        
        self.trainers = [
            ModelTrainer(
                model=self.models[i], 
                device=device, 
                learning_rate=learning_rate, 
                patience=patience
            )
            for i in range(ensemble_size)
        ]

    def train(self, epochs=50, model_dir='ensemble_models'):
        histories = []

        for idx, trainer in enumerate(self.trainers):
            print(f"\n🚀 Training ensemble member {idx+1}/{self.ensemble_size}")
            model_path = f"{model_dir}/ensemble_model_{idx}.pth"
            history = trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=epochs,
                model_path=model_path
            )
            histories.append(history)
        
        return histories

    def predict(self, loader):
        all_member_probs     = []
        all_member_preds     = []
        all_member_max_probs = []
        all_labels_list      = []
        
        for idx, model in enumerate(self.models):
            print(f"\n🔎 Predicting with ensemble member {idx+1}/{self.ensemble_size}")
            model.eval().to(self.device)

            member_probs     = []
            member_preds     = []
            member_max_probs = []

            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"Member {idx+1}", leave=False):
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    max_probs = probs.max(dim=1)[0]

                    member_probs.append(probs.cpu().numpy())
                    member_preds.append(preds.cpu().numpy())
                    member_max_probs.append(max_probs.cpu().numpy())

                    if idx == 0:
                        all_labels_list.append(labels.cpu().numpy())

            all_member_probs.append(np.concatenate(member_probs, axis=0))
            all_member_preds.append(np.concatenate(member_preds, axis=0))
            all_member_max_probs.append(np.concatenate(member_max_probs, axis=0))

        all_member_probs     = np.stack(all_member_probs,     axis=0)  
        all_member_preds     = np.stack(all_member_preds,     axis=0)  
        all_member_max_probs = np.stack(all_member_max_probs, axis=0)  
        all_labels           = np.concatenate(all_labels_list, axis=0)  

        ensemble_probs = all_member_probs.mean(axis=0)            
        ensemble_preds = np.argmax(ensemble_probs, axis=1)         

        max_over_models = all_member_max_probs.max(axis=0)         
        min_over_models = all_member_max_probs.min(axis=0)         
        range_uncertainty = max_over_models - min_over_models      

        return {
            "ensemble_probs":          ensemble_probs,
            "ensemble_preds":          ensemble_preds,
            "all_labels":              all_labels,
            "member_preds":            all_member_preds,
            "variance_across_members": range_uncertainty
        }


    def evaluate(self):
        results = self.predict(self.test_loader)
        metrics = compute_metrics(
            results["all_labels"],
            results["ensemble_preds"],
            results["ensemble_probs"]
        )
        metrics["mean_variance"] = results["variance_across_members"].mean()
        metrics['y_preds'] = results["ensemble_preds"]
        metrics['y_trues'] = results["all_labels"]
        metrics['y_probs'] = results["ensemble_probs"]
        metrics['variance'] = results["variance_across_members"]

        return metrics
    
    def load(self, model_paths):
        for i, model_path in enumerate(model_paths):
            self.models[i].load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.models[i].to(self.device)
            self.models[i].eval()
            print(f"Model {i+1} loaded from {model_path}")
        