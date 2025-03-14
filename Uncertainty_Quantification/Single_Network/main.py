import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from models import EDLModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
print(f"📌 Project root added to PYTHONPATH: {PROJECT_ROOT}")
from Preprocessing.preprocessing import load_extracted_features


FEATURES_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")


def main():

    parser = argparse.ArgumentParser(description="Training an uncertainty-aware classifier on extracted features.")
    parser.add_argument("--train" , action="store_true", help="Train the model")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--dropout", action="store_true", help="Use dropout.")
    parser.add_argument("--uncertainty", action="store_true", help="Enable uncertainty-aware classification.")
    parser.add_argument("--extractor_name", default="ResNet50", type=str, help="Name of the feature extractor.")
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", action="store_true", help="Use Expected Mean Square Error loss.")
    uncertainty_type_group.add_argument("--digamma", action="store_true", help="Use Expected Cross Entropy loss.")
    uncertainty_type_group.add_argument("--log", action="store_true", help="Use Negative Log of the Expected Likelihood loss.")

    args = parser.parse_args()
    device = get_device()
    num_classes = 5
    
    X_train, y_train = load_extracted_features(args.extractor_name, "train")
    X_val, y_val = load_extracted_features(args.extractor_name, "val")
    input_size = X_train.shape[1]
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    dataloaders = {"train": train_loader, "val": val_loader}
    
    
    for tensor in [X_train, y_train, X_val, y_val]:
        tensor = tensor.to(device)


    model = EDLModel(input_size, num_classes).to(device)
    
    if args.train:
        print(f"Training Model (Uncertainty: {args.uncertainty})")
        if args.uncertainty:
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
                raise ValueError("--uncertainty requires --mse, --log, or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()
            model_path = os.path.join(MODEL_SAVE_PATH, f"{args.extractor_name}_baseline.pt")
            
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
            args.uncertainty,
        )
        torch.save({
            "model_state_dict": model.state_dict(),  
            "optimizer_state_dict": optimizer.state_dict(),
            "training_metrics": metrics 
        }, model_path)

        print(f"Model saved to {model_path}")
        
        
if __name__ == "__main__":
    main()
    