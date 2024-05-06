import torch
import os 
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.optim as optim

# Models
from diffusion import GaussianDiffusion
from unet_SR3 import UNet

from tqdm import tqdm



# # *************************
# # STEP 1: MODELS 
# # *************************

# Additional Parameters
device = 'cpu'
img_size = 128
   
# Define parameters of the U-Net (denoising function)
in_channels = 1*2                        
out_channels = 1                        # Output will also be GrayScale
inner_channels = 32                     # Depth feature maps, model complexity 
norm_groups = 32                        # Granularity of normalization, impacting convergence
channel_mults = (1, 2, 4, 8, 8)
attn_res = [8]
res_blocks = 3
dropout = 0
with_noise_level_emb = True
image_size = img_size

# # Instantiate the UNet model
denoise_fn = UNet(
    in_channel=in_channels,
    out_channel=out_channels,
    inner_channel=inner_channels,
    norm_groups=norm_groups,
    channel_mults=channel_mults,
    attn_res=attn_res,
    res_blocks=res_blocks,
    dropout=dropout,
    with_noise_level_emb=with_noise_level_emb,
    image_size=image_size
)

# Define diffusion model parameters
image_size = (img_size, img_size)     # Resized image size
channels = 1             
loss_type = 'l1'
conditional = True          # Currently, the implementation only works conditional

# Noise Schedule from: https://arxiv.org/pdf/2306.01875.pdf
config_diff = {
    'beta_start': 0.0001,
    'beta_end': 0.5,
    'num_steps': 10,      # Reduced number of steps
    'schedule': "quad"    # Should be good with small image sizes
}

# Initialize the Diffusion model 
model = GaussianDiffusion(
    denoise_fn=denoise_fn,
    image_size=image_size,
    channels=channels,
    loss_type=loss_type,
    conditional=conditional,
    config_diff=config_diff
)

# Move models to the GPU if available
denoise_fn.to(device)
model.to(device)

# Info
print('STATUS: Model loaded on device ', device)


# # *************************
# # STEP 2: DATASETS  
#   adapted from:  https://discuss.pytorch.org/t/loading-huge-data-functionality/346/3
# # *************************
class MyDataset(Dataset):
    def __init__(self, data_dir, file_format='pkl'):
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
        self.data_dir = data_dir
        self.file_format = file_format

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # TODO:
            print('Pickle..')
            data = data.to(dtype=torch.float32)
        return data

    def __len__(self):
        return len(self.data_files)

# Example usage:
clean_data_dir = 'gaf_clean_data'
noisy_data_dir = 'gaf_noisy_data'
clean_dataset = MyDataset(clean_data_dir, file_format='pkl')
noisy_dataset = MyDataset(noisy_data_dir, file_format='pkl')

print('len...', len(clean_dataset))
print('len...', len(noisy_dataset))


# DataLoader parameters
batch_size = 32
shuffle = True
num_workers = 1

# Create DataLoader objects
clean_data_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=shuffle)
noisy_data_loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=shuffle)


# # *************************
# # STEP 3: Training
# # *************************

print('Training...')
# Training Parameters
num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Initialize tqdm with the total number of batches
    progress_bar = tqdm(zip(clean_data_loader, noisy_data_loader), total=len(clean_data_loader))
    total_loss = 0.0

    # Iterate over minibatches
    for clean_batch, noisy_batch in progress_bar:
        # Ensure the tensors are on GPU if available
        clean_batch = clean_batch.to(device)
        noisy_batch = noisy_batch.to(device)

        # Forward pass: compute loss
        loss = model({'HR': clean_batch, 'SR': noisy_batch})
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Update progress bar description with the current loss
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Update total loss
        total_loss += loss.item()

    # Calculate average loss for the epoch
    epoch_loss = total_loss / len(clean_data_loader)

    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")
