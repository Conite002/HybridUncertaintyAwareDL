import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from tqdm import tqdm


import time
import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from helpers import get_device, one_hot_embedding
from losses import relu_evidence

def train_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None, uncertainty=False):
    """
    Trains a model using extracted features with uncertainty-aware classification.

    Returns:
    - model: The trained PyTorch model.
    - metrics: A dictionary containing loss and accuracy per epoch.
    """
    since = time.time()
    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics = {"loss": [], "accuracy": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

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

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects / total  

        metrics["loss"].append(epoch_loss)
        metrics["accuracy"].append(epoch_acc)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()  

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Best Train Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    print("\n Training Metrics:")
    for i in range(len(metrics["loss"])):
        print(f"Epoch {i+1}: Loss = {metrics['loss'][i]:.4f}, Accuracy = {metrics['accuracy'][i]:.4f}")

    return model, metrics

        