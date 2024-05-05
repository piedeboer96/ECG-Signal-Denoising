import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import device
from sklearn.model_selection import train_test_split
from datetime import datetime

from diffusion import GaussianDiffusion
from unet import UNet
from embedding import EmbeddingGGM

# CPU/CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Intialize device,', device)

# # *************************
# # STEP 1: MODEL
# # *************************

# Define parameters of the U-Net (denoising function) - Taken from SR3
in_channels = 6                        
out_channels = 3                        # Output will also be GrayScale
inner_channels = 64                     # Depth feature maps, model complexity 
norm_groups = 32                        # Granularity of normalization, impacting convergence
channel_mults = (1, 2, 4, 8, 8)
attn_res = [16]
res_blocks = 3
dropout = 0.2
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
channels = 3             
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

# DATASET 
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

# EMBEDDING 
embedding_ggm = EmbeddingGGM()

# Take subsets for x slices (life is not perfect)
subset_size = 50
for i in range(0, 55000, subset_size):
    
    # Device (return to CUDA after inference if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on device', device)

    # TIMESTAMP
    hour, minute = datetime.now().hour, datetime.now().minute
    formatted_time = f"{hour}h{minute:02d}"

    # SAVE
    save_model_diff = 'diff_model_ggm' + str(formatted_time) + '.pth'
    save_model_dn = 'dn_model_ggm' + str(formatted_time) + '.pth'

    # LOAD SUBSET SLICES
    with open('ardb_slices_clean.pkl', 'rb') as f:
        clean_signals = pickle.load(f)
    clean_signals_subset = clean_signals[i:i+subset_size]
    
    ggm_HR = embedding_ggm.ecg_to_GGM(clean_signals[57000][:128])
    
    del clean_signals       # REMOVE FROM MEMORY

    with open('ardb_slices_noisy.pkl', 'rb') as f:
        noisy_signals = pickle.load(f)

    ggm_SR = embedding_ggm.ecg_to_GGM(noisy_signals[57000][:128])

    noisy_signals_subset= noisy_signals[i:i+subset_size]

    del noisy_signals       # REMOVE FROM MEMORY

    #############################################
    ############################################

    dataset = mijnDataset(clean_signals_subset, noisy_signals_subset)

    # # **************************************
    # # STEP 1b: EMBEDDING AND CUDA
    # # **************************************

    # Initialize lists to store embedded clean and noisy batches
    embedded_clean_batches = []
    embedded_noisy_batches = []
    
    # Iterate through the dataset to embed each sample
    for clean_signal, noisy_signal in dataset:

        clean_ggm = embedding_ggm.ecg_to_GGM(clean_signal)
        noisy_ggm = embedding_ggm.ecg_to_GGM(noisy_signal)

        # Assuming clean_ggm and noisy_ggm are 2D tensors with shape [height, width]
        # Add channel dimension with size 1
        clean_ggm = clean_ggm.unsqueeze(0)  # Adds channel dimension at the beginning
        noisy_ggm = noisy_ggm.unsqueeze(0)  # Adds channel dimension at the beginning

        embedded_clean_batches.append(clean_ggm)
        embedded_noisy_batches.append(noisy_ggm)

    # Convert the lists to torch tensors
    embedded_clean_data = torch.cat(embedded_clean_batches, dim=0)
    embedded_noisy_data = torch.cat(embedded_noisy_batches, dim=0)

    # # **************************************
    # # STEP 2: TRANING
    # # **************************************

    # Example DataLoader creation
    batch_size = 4
    num_workers = 0  # Set according to your system capabilities
    shuffle = True
    # Use the embedded data for training
    dataloader = DataLoader(mijnDataset(embedded_clean_data, embedded_noisy_data), 
                            batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set up your training loop
    num_epochs = 1

    # Initialize best_loss and best_model_state_dict
    best_loss = float('inf')
    best_model_state_dict = None

    # TQDM
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Initialize tqdm for the epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (clean_batch, noisy_batch) in enumerate(pbar):
            # Transfer batches to device if necessary (not needed if you've already done it above)
            clean_batch = clean_batch
            noisy_batch = noisy_batch

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
        	
            # Update progress bar description with current loss
            pbar.set_postfix({'Loss': loss.item()})


        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Check if current model is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state_dict = model.state_dict()

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # ********************
    # SAVE 

    print('Status: Saving Models')
    torch.save(model.state_dict(), save_model_diff)                 # difffusion model
    torch.save(model.denoise_fn.state_dict(), save_model_dn)        # denoising model

# *********************************
print('Status: Finished Training')