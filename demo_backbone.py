import torch
from PIL import Image
from torchvision.transforms import v2

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

backbone = torch.hub.load(DINOV3_REPO_PATH, "dinov3_vitl16", source="local", weights=f"{TORCH_HUB_PATH}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", in_chans=3, use_fa3=False)

print(f"Backbone architecture:\n\n{backbone}")
print(f"=========================================================================")

backbone.eval()

# we inflate 2d image to 3d 
img_size = 2048  # size of 2d dims
D = 2048  # size of new dim
img  = get_img()
transform = make_transform(img_size)

with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        transformed_img = transform(img)[None]
        print(f"transformed_img shape: {transformed_img.shape}")

        B, C, H, W = transformed_img.shape
        inflated_img = transformed_img  # .unsqueeze(2).expand(B, C, D, H, W)  # expand is more memory efficient than repeat
        print(f"inflated_img shape: {inflated_img.shape}")

        # preds = backbone(inflated_img)  # model predictions 
        # spatial_preds = backbone.get_intermediate_layers(inflated_img, n=1, reshape=True, return_class_token=False)  # feature maps
        features = backbone.forward_features(inflated_img)  # final feature map
        # print(f"spatial preds (vitl) len: {len(spatial_preds)}")
        # print(f"spatial preds (vitl) shape: {spatial_preds[0].shape}")
        print(f"x_norm_patchtokens (vitl) shape: {features['x_norm_patchtokens'].shape}")
        print(f"x_prenorm (vitl) shape: {features['x_prenorm'].shape}")        
        # print(f"preds (vitl) shape: {preds.shape}")