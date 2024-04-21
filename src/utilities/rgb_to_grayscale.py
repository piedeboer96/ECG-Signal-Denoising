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

# Adapted to work with grayscale..

# Load CIFAR-10 data
clean_images_train = torch.tensor(cifar10_train.data).permute(0, 3, 1, 2).float() / 255
clean_images_test = torch.tensor(cifar10_test.data).permute(0, 3, 1, 2).float() / 255

# Convert RGB images to grayscale
# def rgb_to_grayscale(images):
#     # Grayscale conversion formula: Y = 0.299*R + 0.587*G + 0.114*B
#     grayscale_images = (images[:, 0, :, :] * 0.299 + images[:, 1, :, :] * 0.587 + images[:, 2, :, :] * 0.114).unsqueeze(1).clone().detach()
#     return grayscale_images

# clean_images_train_gray = rgb_to_grayscale(clean_images_train)
# clean_images_test_gray = rgb_to_grayscale(clean_images_test)

# Add Gaussian noise to grayscale images and make a copy
def add_gaussian_noise(images, mean=0, std=0.1):
    noisy_images = images.clone()
    noisy_images += torch.randn_like(images) * std + mean
    return noisy_images

noisy_images_train = add_gaussian_noise(clean_images_train)
noisy_images_test = add_gaussian_noise(clean_images_test)

# Organize the images into dictionaries
x_in_train = {'HR': clean_images_train, 'SR': noisy_images_train}
x_in_test = {'HR': clean_images_test, 'SR': noisy_images_test}

# # Visualize the images
# def visualize_images(images_hr, images_sr, num_images=5):
#     fig, axes = plt.subplots(2, num_images, figsize=(12, 4))

#     for i in range(num_images):
#         ax1 = axes[0, i]
#         ax2 = axes[1, i]
        
#         ax1.imshow(images_hr[i].squeeze(), cmap='gray')
#         ax1.set_title('HR Image (Grayscale)')
#         ax1.axis('off')

#         ax2.imshow(images_sr[i].squeeze(), cmap='gray')
#         ax2.set_title('SR Image with Gaussian Noise (Grayscale)')
#         ax2.axis('off')

#     plt.tight_layout()
#     plt.show()

# # Visualize the images
# visualize_images(x_in_train['HR'], x_in_train['SR'])

print('Status: Data Loaded Successfully')
