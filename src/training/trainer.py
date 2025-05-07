import torch
from torch import nn
import  torch.optim as optim
import copy
from tqdm import tqdm


class ModelTrainer():
    def __init__(self, model, learning_rate=1e-5, patience=10, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.patience = patience
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate, 
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=self.patience, verbose=True
        )
        

    def train(self, train_loader, val_loader, epochs=50, model_path="best_model.pth"):
        best_val_loss = float('inf')
        epochs_no_improvement = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }       
        
        for epoch in range(epochs):
            self.model.train()
            running_loss, running_corrects = 0.0, 0
                    
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            history['train_loss'].append(epoch_loss.item())
            history['train_acc'].append(epoch_acc.item())
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            
            val_loss, val_acc = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improvement = 0
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), model_path)
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        self.model.load_state_dict(best_model_wts)
        return history
    
    def validate(self, val_loader, step="Validation"):
        self.model.eval()
        running_loss, running_corrects = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds =torch.max(outputs, 1)
                runnning_loss +=loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        print(f"{step} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        return epoch_loss, epoch_acc
