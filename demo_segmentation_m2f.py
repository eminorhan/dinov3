import torch
from PIL import Image
from torchvision.transforms import v2
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.inference import make_inference
from functools import partial

# helper functions
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

# change the following vars according to your setup
TORCH_HUB_PATH = "/lustre/blizzard/stf218/scratch/emin/torch_hub"  # this is where the dinov3 pth checkpoints are stored
DINOV3_REPO_PATH = "/lustre/blizzard/stf218/scratch/emin/dinov3"  # dinov3 repo path

torch.hub.set_dir(TORCH_HUB_PATH)
backbone = torch.hub.load(DINOV3_REPO_PATH, "dinov3_vitl16", source="local", weights=f"{TORCH_HUB_PATH}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", in_chans=3)
model = build_segmentation_decoder(backbone, decoder_type="m2f", num_classes=150)

print(f"Segmentation model architecture:\n\n{model}")
print(f"=========================================================================")

img_size = 896
img  = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        print(f"batch_img shape: {batch_img.shape}")
        preds = model(batch_img)  # raw predictions (keys: 'pred_logits', 'pred_masks', 'aux_outputs')
        print(f"pred_logits: {preds['pred_logits'].shape}")
        # actual segmentation map
        segmentation_map = make_inference(
            batch_img,
            model,
            inference_mode="slide",
            decoder_head_type="m2f",
            rescale_to=(img.size[-1], img.size[-2]),
            n_output_channels=150,
            crop_size=(img_size, img_size),
            stride=(img_size, img_size),
            output_activation=partial(torch.nn.functional.softmax, dim=1),
        ).argmax(dim=1, keepdim=True)
        print(f"segmentation_map shape: {segmentation_map.shape}")
