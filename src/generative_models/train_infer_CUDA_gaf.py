import matplotlib.pyplot as plt
import torch
import pickle
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
device = 'cuda'

# # *************************
# # STEP 1: MODELS 
# # *************************
   
# Define parameters of the U-Net (denoising function)
in_channels = 1*2                        
out_channels = 1                        # Output will also be GrayScale
inner_channels = 32                     # Depth feature maps, model complexity 
norm_groups = 32                            # Granularity of normalization, impacting convergence
channel_mults = (1, 2, 4, 8, 8)
attn_res = [8]
res_blocks = 3
dropout = 0
with_noise_level_emb = True
image_size = 512

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
image_size = (512, 512)     # Resized image size
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

# Embedding Spectrogram 
embedding_gaf = EmbeddingGAF()

# Load Clean and Noisy Slices
with open('slices_clean.pkl', 'rb') as f:
    slices_clean = pickle.load(f)

with open ('slices_noisy.pkl', 'rb') as f:
    slices_noisy = pickle.load(f)



# Larger K means less data...
x_in_train, x_in_test = embedding_gaf.build_gaf_data(clean_slices= slices_clean, noisy_slices=slices_noisy,k=1500)
print('Size of x_in_train', len(x_in_train))
print('Size of x_in_test', len(x_in_test))


del slices_noisy
del slices_clean

# print(x_in_train['SR'][0].shape)
# print(x_in_train['SR'][0].dtype)
# embedding_gaf.visualize_tensor(x_in_train['HR'][0])
# embedding_gaf.visualize_tensor(x_in_train['SR'][0])


# Copy


# # **************************************
# # STEP 2: TRANING
# # **************************************

# Training Config
config_train = {            ## check this...
    'feats':40,
    'epochs':1,
    'batch_size':2,
    'lr':1.0e-3
}

train_model = 1
save_model = 1

# TIMESTAMP
current_time = datetime.now()
hour = current_time.hour
minute = current_time.minute
formatted_time = f"{hour}h{minute:02d}"  # :02d ensures that minutes are displayed with leading zero if less than 10
print(formatted_time)  # Output will be something like: 14h11

# SAVE
save_model_diff = 'diff_model_gaf' + str(formatted_time) + '.pth'
save_model_dn = 'dn_model_gaf' + str(formatted_time) + '.pth'

# Training
if train_model == 1: 

    # Define custom dataset class
    class DictDataset(Dataset):
        def __init__(self, data_dict):
            self.data_dict = data_dict
            self.keys = list(data_dict.keys())

        def __len__(self):
            return len(self.data_dict[self.keys[0]])

        def __getitem__(self, index):
            return {k: v[index] for k, v in self.data_dict.items()}

    # Training Configuration
    feats = config_train['feats']
    epochs = config_train['epochs']
    batch_size = config_train['batch_size']
    lr = config_train['lr']

    # Use custom dataset class for training data
    train_dataset = DictDataset(x_in_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Define DataLoader for testing dataset
    test_dataset = DictDataset(x_in_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on device', device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    print('Status: Training Model')

    best_loss = float('inf')  # Initialize the best loss as positive infinity

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Create tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch_data in pbar:
            # Move tensors to the GPU if available
            batch_data = {key: val.to(device) for key, val in batch_data.items()}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(batch_data)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate total loss
            total_loss += loss.item()

            # Update progress bar description with current loss
            pbar.set_postfix({'Loss': loss.item()})

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Check if current model is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state_dict = model.state_dict()

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

# Save model...  
if save_model ==  1: 
   
    print('Status: Saving Models')

    # Save diffusion model (model)
    torch.save(model.state_dict(), save_model_diff)

    # Save denoising model (UNet) (denoise_fn)
    torch.save(model.denoise_fn.state_dict(), save_model_dn)

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
    image_size=512
).to(device)  # Move the denoising model to the GPU if available

denoise_fun.load_state_dict(torch.load(save_model_dn, map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(512,512),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load(save_model_diff, map_location=device))

print('Status: Diffusion and denoising model loaded successfully')

def visualize_tensor(image_tensors, titles=None):
        num_images = len(image_tensors)

        # Check if titles are provided and if their number matches the number of images
        if titles is not None and len(titles) != num_images:
            print("Error: Number of titles does not match the number of images.")
            return

        # Create subplots based on the number of images
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        # Iterate over images and titles to plot them
        for i, (image_tensor, title) in enumerate(zip(image_tensors, titles)):
            ax = axes[i] if num_images > 1 else axes  # Use appropriate subplot
            ax.axis('off')  # Hide axes
            ax.set_title(title) if title else None  # Set subplot title if provided
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            if len(image_np.shape) == 2:  # If grayscale
                ax.imshow(image_np, cmap='gray')
            else:  # If RGB
                ax.imshow(image_np)

        plt.show()

    # Sample Tensor

idx_sampled = 0

sampled_tensor = diffusion.p_sample_loop_single(x_in_train['SR'][idx_sampled])
sampled_tensor = sampled_tensor.unsqueeze(0)

image_tensors= [x_in_train_original['HR'][idx_sampled],x_in_train_original['SR'][idx_sampled],sampled_tensor ]
names = ['Original HR', 'Original SR', 'Sampled Image'] 

# Visualize Results
visualize_tensor(image_tensors,names)

# Save the sampled_tensor as a pickle file... 
with open('sampled_tensor_gaf.pkl','wb') as f:
    pickle.dump(sampled_tensor, f)
        
    #  TODO:
     # - sav.dump(sampled_tensor, f)
