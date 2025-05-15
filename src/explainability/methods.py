import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchcam.methods.gradient import GradCAM
import torch


def get_image_and_cam(model_or_models, image_tensor, device, approach="single", target_layer="layer4", num_passes=5):
    image_tensor = image_tensor.to(device)

    if approach == "single":
        model = model_or_models.to(device).eval()
        cam_extractor = GradCAM(model, target_layer=target_layer)
        output = model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        cam = cam_extractor(pred_class, output)[0].cpu().numpy()

    elif approach == "mc_dropout":
        model = model_or_models.to(device).eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        cams = []
        outputs = []
        
        for _ in tqdm(range(num_passes)):
            cam_extractor = GradCAM(model, target_layer=target_layer)
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            outputs.append(output)
            cam = cam_extractor(pred, output)[0].cpu().numpy()
            cams.append(cam)
        averaged_cam = np.mean(cams, axis=0)
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        pred_class = avg_output.argmax(dim=1).item()
        cam = averaged_cam
            
            
    elif approach == "deep_ensemble":
        models = [m.to(device).eval() for m  in model_or_models ]
        cams = []
        outputs = []
        
        for model in tqdm(models):
            cam_extractor = GradCAM(model, target_layer=target_layer)
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            outputs.append(output)
            cam = cam_extractor(pred, output)[0].cpu().numpy()
            cams.append(cam)
        averaged_cam = np.mean(cams, axis=0)
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        pred_class = avg_output.argmax(dim=1).item()
        cam = averaged_cam
    else:
        raise ValueError("Unsupported approach")
    
    return cam.squeeze(), pred_class


def show_all_gradcams(image_tensor, cams_dict):
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    plt.figure(figsize=(18, 5))
    for i, (approach, (cam, pred_class)) in enumerate(cams_dict.items()):
        plt.subplot(1, 3, i+1)
        plt.imshow(image_np)
        plt.imshow(cam, alpha=0.5, cmap='jet')
        plt.title(f"{approach} (class {pred_class})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
