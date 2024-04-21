import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from diffusion import GaussianDiffusion
from unet_SR3 import UNet




print('Status: Inference Time...')

# Define the file path to load the model
model_path = "saved_model.pth"

# Initialize the model
model = GaussianDiffusion(
    denoise_fn=denoise_fn,
    image_size=image_size,
    channels=channels,
    loss_type=loss_type,
    conditional=conditional,
    config_diff=config_diff
)

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr=config_train['lr'])

# Load the saved model and optimizer state
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Move model to device
model.to(device)

# Other necessary configurations
config_diff = checkpoint['config_diff']
config_train = checkpoint['config_train']

print("Model loaded successfully from:", model_path)

# Perform inference using the p_sample_loop method
inference_results = model.p_sample_loop(x_in_test['SR'][0], continous=True)
print("Inference Completed Successfully!")