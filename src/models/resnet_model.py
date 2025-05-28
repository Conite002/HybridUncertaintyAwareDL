import torch.nn as nn
import torchvision.models as models

def initialize_resnet_sipakmed(num_classes, dropout_rate=0.5, mc_dropout=False, device='cuda'):
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
            
    dropout_layer = nn.Dropout(dropout_rate) if mc_dropout else nn.Identity()
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        dropout_layer,
        nn.Linear(1024, 128),
        nn.ReLU(),
        # dropout_layer,
        nn.Linear(128, num_classes)
    )
    
    return model.to(device)





def initialize_resnet_ovarian(num_classes, dropout_rate=0.5, mc_dropout=False, device='cuda'):
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
        in_features = model.fc.in_features
        # Define new classifier with MC Dropout if enabled
        dropout_layer = nn.Dropout(p=dropout_rate) if mc_dropout else nn.Identity()
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            dropout_layer,
            nn.Linear(1024, 128),
            nn.ReLU(),
            # dropout_layer,
            nn.Linear(128, 5)
        )
    
    return model.to(device)


