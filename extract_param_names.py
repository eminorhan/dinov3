import torch

# Define path to the checkpoint file
checkpoint_path = '/lustre/gale/stf218/scratch/emin/torch_hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'

# Load the checkpoint onto the CPU
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# First, let's see what keys are in the checkpoint dictionary
print(f"Checkpoint keys: {checkpoint.keys()}")

# Common keys for the model's state_dict are 'model', 'model_state_dict', or 'state_dict'.
# Adjust the key name based on what you see in the output above.
if 'model_state_dict' in checkpoint:
    print('model_state_dict')
    state_dict = checkpoint['model_state_dict']
elif 'model' in checkpoint:
    print('model')
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    print('state_dict')
    state_dict = checkpoint['state_dict']
else:
    # If no specific key is found, assume the whole file is the state_dict
    state_dict = checkpoint  # <-- for DINOv3 checkpoints this is the case

# Now, print all the parameter names from the state_dict
print("\nModel parameter names:")
for name, param in state_dict.items():
    # Printing the name and the shape of the tensor is often useful
    print(f"{name}: {param.shape}")