import torch
import torch.nn as nn
import torch.optim as optim         
import os, sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from Uncertainty_Quantification.train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from Uncertainty_Quantification.models import EDLModel
import json
from test import evaluate_model


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
print(f"📌 Project root added to PYTHONPATH: {PROJECT_ROOT}")
from Preprocessing.preprocessing import load_extracted_features
from Preprocessing.extractors import check_class_distribution

FEATURES_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")


import torch
from torch.utils.data import Subset, DataLoader
import numpy as np

def split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.1, cal_ratio=0.1, test_ratio=0.1):
    """
    Réduit le dataset en train (70%), val (10%), cal (10%), test (10%).
    
    Arguments:
    - full_dataset (TensorDataset) : Jeu de données complet.
    - train_ratio (float) : Portion du dataset pour `train` (par défaut 70%).
    - val_ratio (float) : Portion du dataset pour `val` (par défaut 10%).
    - cal_ratio (float) : Portion du dataset pour `cal` (par défaut 10%).
    - test_ratio (float) : Portion du dataset pour `test` (par défaut 10%).

    Retourne:
    - dict contenant `train`, `val`, `cal`, `test` en `Subset`.
    """
    assert train_ratio + val_ratio + cal_ratio + test_ratio == 1, "Les ratios doivent totaliser 100%"
    assert round(train_ratio + val_ratio + cal_ratio + test_ratio, 5) == 1, "Les ratios doivent totaliser 100%"
    total_size = len(full_dataset)
    indices = np.random.permutation(total_size)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    cal_size = int(total_size * cal_ratio)
    test_size = total_size - (train_size + val_size + cal_size)  
    # Séparer les indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    cal_indices = indices[train_size + val_size:train_size + val_size + cal_size]
    test_indices = indices[train_size + val_size + cal_size:]

    # Création des sous-ensembles
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    cal_dataset = Subset(full_dataset, cal_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "cal": cal_dataset,
        "test": test_dataset
    }




def main():

    parser = argparse.ArgumentParser(description="Training an uncertainty-aware classifier on extracted features.")
    parser.add_argument("--train" , action="store_true", help="Train the model")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--dropout", action="store_true", help="Use dropout.")
    parser.add_argument("--uncertainty_method", default="single", help="Enable uncertainty-aware classification.")
    parser.add_argument("--extractor_name", default="ResNet50", type=str, help="Name of the feature extractor.")
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", action="store_true", help="Use Expected Mean Square Error loss.")
    uncertainty_type_group.add_argument("--digamma", action="store_true", help="Use Expected Cross Entropy loss.")
    uncertainty_type_group.add_argument("--log", action="store_true", help="Use Negative Log of the Expected Likelihood loss.")

    args = parser.parse_args()
    device = get_device()
    num_classes = 5
    
    X_train, y_train = load_extracted_features(args.extractor_name, split="train")
    X_cal, y_cal = load_extracted_features(args.extractor_name, split="cal")
    X_val, y_val = load_extracted_features(args.extractor_name, split="val")
    X_test, y_test = load_extracted_features(args.extractor_name, split="test")

    input_size = X_train.shape[1]
    
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    # val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    # cal_dataset = torch.utils.data.TensorDataset(X_cal, y_cal)
    # test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Création du dataset complet (avant réduction)
    full_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # Répartition automatique en 70% train, 10% val, 10% cal, 10% test
    datasets = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.1, cal_ratio=0.1, test_ratio=0.1)

    # Création des DataLoader
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    cal_loader = DataLoader(datasets["cal"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    dataloaders = {"train": train_loader, "val": val_loader, "cal": cal_loader, "test": test_loader}

    dataloaders = {"train": train_loader, "val": val_loader, "cal": cal_loader, "test": test_loader}

    
    for tensor in [X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test]:
        tensor = tensor.to(device)


    model = EDLModel(input_size, num_classes).to(device)
    
    if args.train:
        print(f"Training Model (Uncertainty: {args.uncertainty_method})")
        if args.uncertainty_method == "deep_ensemble":
            if args.mse:
                criterion = edl_mse_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_mse.pt")
            elif args.digamma:
                criterion = edl_digamma_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_digamma.pt")
            elif args.log:
                criterion = edl_log_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_log.pt")
            else:
                raise ValueError("--uncertainty_method=deep_ensemble requires --mse, --log, or --digamma.")

        elif args.uncertainty_method == "mc_dropout":
            if args.mse:
                criterion = edl_mse_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_mse.pt")
            elif args.digamma:
                criterion = edl_digamma_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_digamma.pt")
            elif args.log:
                criterion = edl_log_loss
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_log.pt")
            else:
                raise ValueError("--uncertainty_method=mc_dropout requires --mse, --log, or --digamma.")

        elif args.uncertainty_method == "single" or args.uncertainty_method is None:
            criterion = nn.CrossEntropyLoss()
            model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_baseline.pt")

        else:
            raise ValueError("Invalid --uncertainty_method. Choose from 'single', 'deep_ensemble', or 'mc_dropout'.")
            
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler,
            args.epochs,
            device,
            args.uncertainty_method,
        )
        torch.save({
            "model_state_dict": model.state_dict(),  
            "optimizer_state_dict": optimizer.state_dict(),
            "training_metrics": metrics 
        }, model_path)

        print(f"Model saved to {model_path}")
        check_class_distribution(dataloaders["val"], "Validation")

        
        # --- Save Metrics ---
        output_dir = "Outputs/models/"
        os.makedirs(output_dir, exist_ok=True)
        model_name = model_path.split("/")[-1].split(".")[0]
        metrics_path = os.path.join(PROJECT_ROOT, output_dir, f"{model_name}_metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"[INFO] Metrics saved to {metrics_path}")
    
    else:
        print(f"Evaluating Model (Uncertainty: {args.uncertainty_method})")

        if args.uncertainty_method == "deep_ensemble":
            if args.mse:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_mse.pt")
            elif args.digamma:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_digamma.pt")
            elif args.log:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_edl_log.pt")
        elif args.uncertainty_method == "mc_dropout":
            if args.mse:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_mse.pt")
            elif args.digamma:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_digamma.pt")
            elif args.log:
                model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_mc_dropout_log.pt")
        else:
            model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_baseline.pt")

        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model = model.to(args.device)

        test_loss, test_acc, y_true, y_pred = evaluate_model(
            model,
            dataloaders["test"],
            len(set(y_train)), 
            criterion,
            args.device,
            args.uncertainty_method
        )

        print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
    