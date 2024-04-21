import torch
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


def download_cifar10():
    # Download CIFAR-10 dataset
    cifar10_train = CIFAR10(root="./data", train=True, download=True)
    cifar10_test = CIFAR10(root="./data", train=False, download=True)
    return cifar10_train, cifar10_test

# Load CIFAR-10 data
cifar10_train, cifar10_test = download_cifar10()

# Convert to tensor and normalize
clean_images_train = torch.tensor(cifar10_train.data).permute(0, 3, 1, 2).float() / 255
clean_images_test = torch.tensor(cifar10_test.data).permute(0, 3, 1, 2).float() / 255

# Resize the images to 16x16
resize_factor = 16 / 32  # Assuming the original size is 32x32
clean_images_train_resized = TF.resize(clean_images_train, (int(16), int(16)))
clean_images_test_resized = TF.resize(clean_images_test, (int(16), int(16)))

# ************************************************************ 
print('Shape of image...', clean_images_train_resized[0].shape)
# ************************************************************ 

# Function to visualize CIFAR-10 images
def visualize_images(images, title):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize original CIFAR-10 images
visualize_images(clean_images_train, "Original CIFAR-10 Train Images")

# Visualize resized CIFAR-10 images
visualize_images(clean_images_train_resized, "Resized CIFAR-10 Train Images (16x16)")
