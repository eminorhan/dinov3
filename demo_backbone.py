import sys
from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.models import build_segmentation_decoder

def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return v2.Compose([to_tensor, resize, to_float, normalize])

TORCH_HUB_PATH = "/lustre/gale/stf218/scratch/emin/torch_hub"  # this is where the dinov3 pth checkpoints are stored
DINOV3_REPO_PATH = "/lustre/gale/stf218/scratch/emin/dinov3"  # dinov3 repo path

torch.hub.set_dir(TORCH_HUB_PATH)

bbone = torch.hub.load(DINOV3_REPO_PATH, "dinov3_vitl_3D16", source="local", weights=f"{TORCH_HUB_PATH}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

print(f"Backbone architecture:\n{bbone}")
print(f"=========================================================================")

img_size = 512
D = 512  # size of new dim
img  = get_img()
transform = make_transform(img_size)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        transformed_img = transform(img)[None]
        print(f"transformed_img shape: {transformed_img.shape}")
        B, C, H, W = transformed_img.shape
        inflated_img = transformed_img.unsqueeze(2).expand(B, C, D, H, W)  # expand is more memory efficient
        print(f"inflated_img shape: {inflated_img.shape}")
        preds = bbone(inflated_img)  # model predictions 
        print(f"preds (vitl) shape: {preds.shape}")