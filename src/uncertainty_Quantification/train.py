import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import sys
sys.path.append(PROJECT_ROOT)
from utils.utils import load_features, save_history, get_device
from models import SingleNetwork, MCDropoutNetwork, DeepEnsemble


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

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}")

        logging.info(f"{model_name} Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"Validation loss decreased to {val_loss:.4f}. Saving model...")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    torch.save(best_model_state, os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))
    save_history(history, os.path.join(HISTORY_SAVE_PATH, f"{model_name}_history.json"))



def train_deep_ensemble(train_loader, val_loader, ensemble_size=5, learning_rate=0.0005, epochs=30):
    ensemble_models = []

    for i in range(ensemble_size):
        logging.info(f"Training Deep Ensemble Model {i+1}/{ensemble_size}")
        model = SingleNetwork().to(DEVICE)  
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name=f"DeepEnsemble_{i}", patience=5)
        # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"deep_ensemble_{i}.pth"))

        ensemble_models.append(model)

    logging.info("Deep Ensemble Training Complete!")
    return ensemble_models
