import torch
import torch.nn as nn
import torch.optim as optim

# 📌 Single Network Model
class SingleNetwork(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5):
        super(SingleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x  
    
    
# 📌 Monte Carlo Dropout Model
class MCDropoutNetwork(nn.Module):
    def __init__(self, input_dim=2048, num_classes=5, dropout_rate=0.5):
        super(MCDropoutNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 📌 Deep Ensemble Class
class DeepEnsemble:
    def __init__(self, num_models=5, input_dim=2048, num_classes=5, device="cuda"):
        self.device = device
        self.models = [SingleNetwork(input_dim, num_classes).to(device) for _ in range(num_models)]

    def predict(self, dataloader):
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                
                preds = torch.stack([torch.softmax(model(X_batch), dim=1) for model in self.models])
                mean_preds = preds.mean(dim=0)
                uncertainty = preds.var(dim=0).sum(dim=1)

                all_preds.append((mean_preds.cpu().numpy(), uncertainty.cpu().numpy()))

        return all_preds
