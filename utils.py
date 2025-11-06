# utils.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

def tensor_to_image(tensor):
    """Convert torch tensor (C,H,W) in [-1,1] to PIL Image."""
    t = tensor.detach().clamp(-1,1).add(1).div(2).mul(255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(t)

def save_image(tensor, path):
    img = tensor_to_image(tensor)
    img.save(path)

def l2_loss(a,b):
    return (a-b).pow(2).mean()
