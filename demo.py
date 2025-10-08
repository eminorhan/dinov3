import sys
from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference

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

REPO_DIR = "/lustre/gale/stf218/scratch/emin/dinov3"
torch.hub.set_dir("/lustre/gale/stf218/scratch/emin/torch_hub")
# segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local", weights="dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth", backbone_weights="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
segmentor = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_ms', source="local")
print(f"Loaded model...")
print(f"Model architecture: {segmentor}") 

img_size = 896
img  = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        pred_vit7b = segmentor(batch_img)  # raw predictions 
        # actual segmentation map
        segmentation_map_vit7b = make_inference(
            batch_img,
            segmentor,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.axis("off")
plt.subplot(122)
plt.imshow(segmentation_map_vit7b[0,0].cpu(), cmap=colormaps["Spectral"])
plt.axis("off")
plt.savefig(f"demo_seg.jpeg", bbox_inches='tight')