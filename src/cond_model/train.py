import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from diffusion import GaussianDiffusion
from unet_SR3 import UNet


# *********************************************
# Step 1: Load Dataset and Prepare Data

# Download CIFAR-10 dataset
def download_cifar10(data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
    ])
    cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return cifar10_train, cifar10_test

cifar10_train, cifar10_test = download_cifar10()

# Keep a copy of the clean images
clean_images_train = torch.tensor(cifar10_train.data).permute(0, 3, 1, 2).float() / 255
clean_images_test = torch.tensor(cifar10_test.data).permute(0, 3, 1, 2).float() / 255

# Add Gaussian noise to images and make a copy
def add_gaussian_noise(images, mean=0, std=0.1):
    noisy_images = images.clone()
    noisy_images += torch.randn_like(images) * std + mean
    return noisy_images

noisy_images_train = add_gaussian_noise(clean_images_train)
noisy_images_test = add_gaussian_noise(clean_images_test)

# Organize the images into dictionaries
x_in_train = {'HR': clean_images_train, 'SR': noisy_images_train}
x_in_test = {'HR': clean_images_test, 'SR': noisy_images_test}

# Visualize the images
def visualize_images(images_hr, images_sr, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))

    for i in range(num_images):
        ax1 = axes[0, i]
        ax2 = axes[1, i]
        
        ax1.imshow(images_hr[i].permute(1, 2, 0))
        ax1.set_title('HR Image')
        ax1.axis('off')

        ax2.imshow(images_sr[i].permute(1, 2, 0))
        ax2.set_title('SR Image with Gaussian Noise')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Visualize the images
visualize_images(x_in_train['HR'], x_in_train['SR'])

print('Status: Data Loaded Successfully')

# *************************************
# STEP 2: Diffusion Model and  Denoising Function         

# Define parameters of the U-Net (denoising function)
in_channels=6           # Concat x_noisy and y_in results 2 (*3) RGB = 6
out_channels=3          # RGB
inner_channels = 8      # Depth feature maps, model complexity 
norm_groups = 8         # Granularity of normalization, impacting convergence
channel_mults = (1, 2, 4, 8, 8)
attn_res = (8,)
res_blocks = 3
dropout = 0
with_noise_level_emb = True
image_size = 32

# Instantiate the UNet model
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
image_size = (32, 32)  # Resized image size
channels = 3  # CIFAR-10 images are RGB
loss_type = 'l1'
conditional = True          # Currently implementation only works conditional

# Noise Schedule from : https://arxiv.org/pdf/2306.01875.pdf
config_diff = {
    'beta_start':  0.0001,
    'beta_end':  0.02,
    'num_steps': 100,  # Reduced number of steps
    'schedule': "quad"
}

# Intialize the Diffusion  model 
model = GaussianDiffusion(
    denoise_fn=denoise_fn,
    image_size=image_size,
    channels=channels,
    loss_type=loss_type,
    conditional=conditional,
    config_diff=config_diff
)


# *******************************
# *******************************
# Step 3: Train the Model 

# Training Config
config_train = { 
    'feats':20,
    'epochs':1,
    'batch_size':32,
    'lr':1.0e-1
}


training_on = 1

if training_on == 1: 

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model.to(device)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    print('Status: Training Model')

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Create tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch_data in pbar:
            # Move tensors to device
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)

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

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    # Optionally, save the trained model
    # torch.save(model.state_dict(), "denoising_diffusion_model.pth")

    # Save the trained model (and the denoising function paramters)
    torch.save({
        'gaussian_diffusion_state_dict': model.state_dict(),
        'denoise_fn_state_dict': denoise_fn.state_dict()
    }, 'model_checkpoint.pth')


# *******************************
# *******************************
# Step 4: Inference

print('Status: Inference Time...')

# Load the saved model
model = GaussianDiffusion(
    denoise_fn=model,
    image_size=image_size,
    channels=channels,
    loss_type=loss_type,
    conditional=conditional,
    config_diff=config_diff
)

checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['gaussian_diffusion_state_dict'])
denoise_fn.load_state_dict(checkpoint['denoise_fn_state_dict'])
# model.eval()

# TODO: may have to set the denoise_fn ... like this.
#model.denoise_fn=denoise_fn.load_state_dict(checkpoint['denoise_fn_state_dict'])

# Perform inference using the p_sample_loop method
inference_results = model.p_sample_loop(x_in_test['SR'], continous=True)
print("Inference Completed Successfully!")