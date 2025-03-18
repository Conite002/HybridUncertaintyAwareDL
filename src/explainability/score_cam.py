import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import cv2

class ScoreCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize ScoreCAM
        
        Args:
            model: The neural network model
            target_layer: Name of the target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Get the target layer
        target_module = dict([*self.model.named_modules()])[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
        
    def generate_cam(self, 
                    input_image: torch.Tensor,
                    target_class: Optional[int] = None,
                    batch_size: int = 32) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses the model's prediction
            batch_size: Batch size for processing activation maps
            
        Returns:
            Class activation map as a numpy array
        """
        self.model.eval()
        
        # Get model prediction and activations
        with torch.no_grad():
            output = self.model(input_image)
            if isinstance(output, tuple):
                output = output[0]
                
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
        # Get activation maps
        activation_maps = self.activations[0]  # (C, H, W)
        
        # Normalize each activation map and resize to input size
        normalized_maps = []
        for activation_map in activation_maps:
            normalized_map = F.interpolate(activation_map.unsqueeze(0).unsqueeze(0),
                                        size=input_image.shape[2:],
                                        mode='bilinear',
                                        align_corners=False)
            normalized_map = (normalized_map - normalized_map.min()) / (normalized_map.max() - normalized_map.min() + 1e-8)
            normalized_maps.append(normalized_map)
            
        normalized_maps = torch.cat(normalized_maps)  # (C, 1, H, W)
        
        # Process in batches
        scores = []
        num_maps = normalized_maps.shape[0]
        for i in range(0, num_maps, batch_size):
            batch_maps = normalized_maps[i:i+batch_size]
            
            # Element-wise multiplication with input
            masked_input = input_image * batch_maps.unsqueeze(1)
            
            # Get model predictions for masked inputs
            with torch.no_grad():
                batch_outputs = self.model(masked_input)
                if isinstance(batch_outputs, tuple):
                    batch_outputs = batch_outputs[0]
                batch_scores = batch_outputs[:, target_class]
                scores.extend(batch_scores.cpu().numpy())
                
        # Weight activation maps by scores
        weights = torch.tensor(scores).to(input_image.device)
        weights = F.softmax(weights, dim=0)
        
        # Generate weighted sum of activation maps
        cam = torch.zeros_like(normalized_maps[0])
        for i, weight in enumerate(weights):
            cam += weight * normalized_maps[i]
            
        # Post-process CAM
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize
        
        return cam
    
    def overlay_cam(self, 
                   image: np.ndarray,
                   cam: np.ndarray,
                   alpha: float = 0.5,
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay CAM on the input image
        
        Args:
            image: Input image as numpy array (H, W, C)
            cam: Class activation map
            alpha: Transparency factor
            colormap: OpenCV colormap
            
        Returns:
            Visualization with CAM overlaid on input image
        """
        # Resize CAM to image size if needed
        if cam.shape != image.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
            
        # Apply colormap to CAM
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay CAM on image
        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 1 - alpha, cam_heatmap, alpha, 0)
        
        return overlay 