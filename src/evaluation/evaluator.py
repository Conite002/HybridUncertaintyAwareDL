from src.evaluation.metrics import compute_metrics

class Evaluator:
    
    def __init__(self, model, loader=None, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.loader = loader
        
    def evaluate(self):
        probs, preds, labels = self.model.predict(self.loader)
        metrics = compute_metrics(probs, preds, labels)
        
        return metrics