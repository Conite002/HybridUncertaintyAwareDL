import torch
import scipy.ndimage as nd


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=5):
    """
    Converts integer labels into one-hot encoded labels.
    """
    device = labels.device 
    y = torch.eye(num_classes, device=device)  
    return y[labels]


def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()