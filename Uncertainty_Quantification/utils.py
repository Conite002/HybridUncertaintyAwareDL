import torch

def enable_dropout(model):
    """
    Enables Dropout layers at inference time for Monte Carlo Dropout.
    This ensures that Dropout is applied during test-time sampling.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
