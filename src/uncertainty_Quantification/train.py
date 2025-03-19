import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.utils import save_history, setup_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import sys
sys.path.append(PROJECT_ROOT)
from utils.utils import load_features, save_history, get_device
from src.uncertainty_Quantification.models import SingleNetwork, MCDropoutNetwork, DeepEnsemble


# Paths
FEATURES_PATH = os.path.join(PROJECT_ROOT, "outputs/feature_extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/models")
HISTORY_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/results")

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HISTORY_SAVE_PATH, exist_ok=True)
DEVICE = get_device()

# 📌 Get DataLoader
def get_dataloader(split, batch_size):
    X, y = load_features(FEATURES_PATH, split)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" or split == "val"))

# 📌 Train Model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name, patience=5):
    model.to(DEVICE)
    best_val_loss = float("inf") 
    best_model_state = None
    patience_counter = 0 
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_recall": [], "val_precision": []}
    logger = setup_logger(model_name)
    logger.info(f"Starting training for {model_name}")

    print(f"Training {model_name} for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{epochs}"):
        # for X_batch, y_batch in train_loader :
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(y_batch.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
        history["train_acc"].append(train_acc)
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        val_precision = precision_score(val_targets, val_preds, average='weighted')

        history["train_loss"].append(running_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)
        history["val_precision"].append(val_precision)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            logger.info(f"Validation loss decreased to {val_loss:.4f}. Saving model...")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    print(f"Training complete for {model_name}")
    torch.save(best_model_state, os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))
    save_history(history, os.path.join(HISTORY_SAVE_PATH, f"{model_name}_history.json"))



def train_deep_ensemble(train_loader, val_loader, ensemble_size=5, learning_rate=0.0005, epochs=30, models=None, optimizer=None, criterion=None, patience=5):
    
    logger = setup_logger("DeepEnsemble")
    logger.info(f"Starting Deep Ensemble Training with {ensemble_size} models")
    ensemble_models = []

    if models:
        for i in range(len(models)):
            train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name=f"DeepEnsemble_{i}", patience=patience)
            ensemble_models.append(model)
    else:
        for i in range(ensemble_size):
            model = SingleNetwork().to(DEVICE) 
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name=f"DeepEnsemble_{i}", patience=patience)
            ensemble_models.append(model)
            
    logger.info("Deep Ensemble Training Complete!")

    return ensemble_models


