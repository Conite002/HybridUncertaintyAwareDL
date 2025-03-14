import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from tqdm import tqdm
import torch.nn.functional as F
import copy
import json
import os


def train_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None, uncertainty=False):
    """
    Trains a model using extracted features with optional uncertainty-aware classification.

    Returns:
    - model: The trained PyTorch model with the best validation accuracy.
    - metrics: A dictionary containing loss and accuracy per epoch for both training and validation.
    """
    since = time.time()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        data_loader = dataloaders["train"]

        with tqdm(total=len(data_loader), desc="Training Progress") as pbar:
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    if uncertainty:
                        labels = labels.to(device)
                        y = one_hot_embedding(labels, num_classes).to(device)

                        outputs = model(inputs, activation="softplus")
                        print(f"[DEBUG] Batch size: {inputs.shape[0]}")
                        print(f"[DEBUG] Model Output Shape: {outputs.shape}")
                        print(f"[DEBUG] One-hot Labels Shape: {y.shape}")

                        loss = criterion(outputs, y.float(), epoch, num_classes, annealing_step=5, device=device)

                        preds = torch.argmax(outputs, dim=1)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        total_evidence = torch.sum(evidence, dim=1, keepdim=True) + 1e-8
                        mean_evidence = torch.mean(total_evidence)

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

                pbar.update(1)

        epoch_train_loss = running_loss / len(data_loader.dataset)
        epoch_train_acc = running_corrects / total  

        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_accuracy"].append(epoch_train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Accuracy: {epoch_train_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)


                if uncertainty:
                    y = one_hot_embedding(labels, num_classes).to(device)
                    loss = criterion(outputs, y.float(), epoch, num_classes, annealing_step=5, device=device)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / len(dataloaders["val"].dataset)
        epoch_val_acc = val_corrects / val_total  

        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_accuracy"].append(epoch_val_acc)

        print(f"Validation Loss: {epoch_val_loss:.4f} | Validation Accuracy: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()  

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Best Validation Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, metrics