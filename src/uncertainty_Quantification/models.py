import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

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

class DeepEnsemble:
    def __init__(self, num_models=5, input_dim=2048, num_classes=5, device="cuda", other_models=None):
        self.device = device
        self.models = [SingleNetwork(input_dim, num_classes).to(device) for _ in range(num_models)]
        if other_models:
            self.models = other_models
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
    
    
class FineTunedInceptionV3(nn.Module):
    def __init__(self, num_classes, pretrained=True, mc_dropout=False, dropout_rate=0.5):
        super(FineTunedInceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=True)
        self.dropout = nn.Dropout(dropout_rate) if mc_dropout else nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.model(x))
        return self.sigmoid(x)

class FineTunedEfficientNetB7(nn.Module):
    def __init__(self, num_classes, pretrained=True, mc_dropout=False, dropout_rate=0.5):
        super(FineTunedEfficientNetB7, self).__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout_rate) if mc_dropout else nn.Identity()
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.model(x))
        return self.sigmoid(x)


class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True, mc_dropout=False, dropout_rate=0.5):
        super(FineTunedResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate) if mc_dropout else nn.Identity(),
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    
    import os, sys
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    sys.path.append(PATH)
    from utils.utils import get_dataloaders
    from src.uncertainty_Quantification.train import train_model, train_deep_ensemble
    from src.uncertainty_Quantification.test import evaluate_model, monte_carlo_dropout, evaluate_mc_dropout
    
    dropout_rate = 0.4
    learning_rate = 0.001
    num_epochs = 50
    num_samples, ensemble_size = 10, 10
    train_loader, val_loader, cal_loader, test_loader = get_dataloaders( batch_size=32, num_workers=4, model_type="resnet")
    
    model = FineTunedResNet50(num_classes=5, mc_dropout=False, dropout_rate=dropout_rate)
    model_path = PATH + "/outputs/models/SingleNetwork_finetuned_resnet50.pth"
    model.load_state_dict(torch.load(model_path))
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    # train_model(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, model_name="SingleNetwork_finetuned_resnet50")
    res_single_resnet50 = evaluate_model(model, test_loader, model)
    print(f"Single Network ResNet50 Test Performance: {res_single_resnet50}")

    # monte carlo dropout
    # model = FineTunedResNet50(num_classes=5, mc_dropout=True, dropout_rate=dropout_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    # train_model(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, model_name="MCDropout_finetuned_resnet50")
    # res_mc_resnet50 = evaluate_mc_dropout(model, test_loader, criterion, num_samples=num_samples)
    # print(f"MC Dropout ResNet50 Test Performance: {res_mc_resnet50}")