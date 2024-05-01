import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import device
from sklearn.model_selection import train_test_split
from datetime import datetime

# Models
from diffusion import GaussianDiffusion
from unet_SR3 import UNet

# Embedding
from embedding_gaf import EmbeddingGAF

# # Check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def plot_multiple_timeseries(signals, names):
    num_signals = len(signals)
    
    plt.figure(figsize=(5 * num_signals, 4))

    for i, (signal, name) in enumerate(zip(signals, names), 1):
        plt.subplot(1, num_signals, i)
        plt.plot(signal)
        plt.title(name)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()




# # *************************
# # STEP 1: MODELS 
# # *************************
   
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
image_size = 128

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

# # Define diffusion model parameters
image_size = (128, 128)     # Resized image size
channels = 1             
loss_type = 'l1'
conditional = True          # Currently, the implementation only works conditional

# # Noise Schedule from: https://arxiv.org/pdf/2306.01875.pdf
config_diff = {
    'beta_start': 0.0001,
    'beta_end': 0.5,
    'num_steps': 10,      # Reduced number of steps
    'schedule': "quad"
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

# # Move models to the GPU if available
denoise_fn.to(device)
model.to(device)

print('STATUS --- Model loaded on device:', device)

# # **************************************
# # STEP 2: DATA LOADING (SMALL)
# # **************************************

# TIMESTAMP
hour, minute = datetime.now().hour, datetime.now().minute
formatted_time = f"{hour}h{minute:02d}"

# SAVING MODEL
# save_model_diff = 'diff_model_gaf' + str(formatted_time) + '.pth'
# save_model_dn = 'dn_model_gaf' + str(formatted_time) + '.pth'

# EMBEDDING GAFF
embedding_gaf = EmbeddingGAF()

# # LOAD CLEAN AND NOISY SLICES
# with open('slices_clean_fs_128.pkl', 'rb') as f:
#     clean_signals = pickle.load(f)

# clean_signals = clean_signals[:50000]

# with open('slices_noisy_EM_snr_3_fs_128.pkl', 'rb') as f:
#     noisy_signals = pickle.load(f)

# noisy_signals = noisy_signals[:50000]



### CUSTOM DATASET
class mijnDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        
    def __len__(self):
        return len(self.clean_signals)
    
    def __getitem__(self, index):
        clean_signal = torch.tensor(self.clean_signals[index], dtype=torch.float32)
        noisy_signal = torch.tensor(self.noisy_signals[index], dtype=torch.float32)
        return clean_signal, noisy_signal

# dataset = mijnDataset(clean_signals, noisy_signals)

# # **************************************
# # STEP 1b: EMBEDD THE DATA
# # **************************************


# TIMESTAMP
hour, minute = datetime.now().hour, datetime.now().minute
formatted_time = f"{hour}h{minute:02d}"

# SAVING MODEL
save_model_diff = 'diff_model_gaf' + str(formatted_time) + '.pth'
save_model_dn = 'dn_model_gaf' + str(formatted_time) + '.pth'

# EMBEDDING GAFF
embedding_gaf = EmbeddingGAF()

# LOAD CLEAN AND NOISY SLICES
with open('slices_clean_fs_128.pkl', 'rb') as f:
    clean_signals = pickle.load(f)

with open('slices_noisy_EM_snr_3_fs_128.pkl', 'rb') as f:
    noisy_signals = pickle.load(f)

num_samples = len(clean_signals)

# Define batch size and batch count
batch_size = 1000
batch_count = int(np.ceil(num_samples / batch_size))

# Initialize memory-mapped arrays to store embedded clean and noisy data
clean_file = 'clean_embedded.dat'
noisy_file = 'noisy_embedded.dat'

clean_memmap = np.memmap(clean_file, dtype='float32', mode='w+', shape=(num_samples, 1, 128, 128))
noisy_memmap = np.memmap(noisy_file, dtype='float32', mode='w+', shape=(num_samples, 1, 128, 128))

# Iterate through batches to embed the data and store it in memory-mapped arrays
for i in range(batch_count):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)

    clean_batch = clean_signals[start_idx:end_idx]
    noisy_batch = noisy_signals[start_idx:end_idx]

    embedded_clean_batch = np.array([embedding_gaf.ecg_to_GAF(signal) for signal in clean_batch])
    embedded_noisy_batch = np.array([embedding_gaf.ecg_to_GAF(signal) for signal in noisy_batch])

    clean_memmap[start_idx:end_idx, 0] = embedded_clean_batch  # Assign embedded_clean_batch directly to clean_memmap
 # Remove extra singleton dimension
    noisy_memmap[start_idx:end_idx] = embedded_noisy_batch.squeeze(1)  # Remove extra singleton dimension


# Create datasets from memory-mapped arrays
clean_dataset = torch.utils.data.TensorDataset(torch.tensor(clean_memmap), torch.tensor(noisy_memmap))

# Example DataLoader creation
num_workers = 0  # Set according to your system capabilities
shuffle = True
dataloader = DataLoader(clean_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Set up your training loop
num_epochs = 2

# TQDM
for epoch in range(num_epochs):
    total_loss = 0.0

    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch_idx, (clean_batch, noisy_batch) in pbar:
            # Transfer batches to device if necessary (not needed if you've already done it above)
            clean_batch = clean_batch.to(device)
            noisy_batch = noisy_batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model({'HR': clean_batch, 'SR': noisy_batch})  # Assuming input format is {'HR': clean, 'SR': noisy}

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

            # Optionally print loss statistics
            if batch_idx % 5 == 0:
                avg_loss = total_loss / 100
                pbar.set_postfix({'Loss': avg_loss})
                
                # Reset total_loss after updating the progress bar
                total_loss = 0.0

# ********************
# SAVE 

print('Status: Saving Models')
torch.save(model.state_dict(), save_model_diff)                 # difffusion model
torch.save(model.denoise_fn.state_dict(), save_model_dn)        # denoising model

    
# *************************************************
# Step 4: Inference (or continue training)

print('Status: Inference Time...')

# ! INFERENCE NEEDS TO BE ON CPU
inference_device = 'cpu'
device = inference_device

# Load a trained denoiser...
denoise_fun = UNet(
    in_channel=2,
    out_channel=1,
    inner_channel=inner_channels,
    norm_groups=norm_groups,
    channel_mults=channel_mults,
    attn_res=[8],
    res_blocks=res_blocks,
    dropout=dropout,
    with_noise_level_emb=with_noise_level_emb,
    image_size=128
).to(device)  # Move the denoising model to the GPU if available

denoise_fun.load_state_dict(torch.load(save_model_dn, map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load(save_model_diff, map_location=device))

print('Status: Diffusion and denoising model loaded successfully')

idx_sampled = 666

#### RANDOM SAMPLE FOR INFERENCE
original_HR = embedding_gaf.ecg_to_GAF(clean_signals[idx_sampled])
noisy_SR = embedding_gaf.ecg_to_GAF(noisy_signals[idx_sampled])

sampled_tensor = diffusion.p_sample_loop_single(noisy_SR)
sampled_tensor = sampled_tensor.unsqueeze(0)

# Save the sampled_tensor as a pickle file... 
save_tensor_sample = 'sampled_tensor_CPU' + str(formatted_time)
with open(save_tensor_sample,'wb') as f:
    pickle.dump(sampled_tensor, f)

