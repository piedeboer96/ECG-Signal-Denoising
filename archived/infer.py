import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import device

from diffusion import GaussianDiffusion
from unet_SR3 import UNet

# *********************************************
# Step 1: Load Dataset and Prepare Data

load_data = 1

if load_data==1:

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

    # Load CIFAR-10 data
    clean_images_train = torch.tensor(cifar10_train.data).permute(0, 3, 1, 2).float() / 255
    clean_images_test = torch.tensor(cifar10_test.data).permute(0, 3, 1, 2).float() / 255

    # Cut down the size of train/test drastically 
    clean_images_train = clean_images_train[:200]
    clean_images_test = clean_images_test[:40]
    print(len(clean_images_train))
    print(len(clean_images_test))

    # Add Gaussian noise to grayscale images and make a copy
    def add_gaussian_noise(images, mean=0, std=0.2):            # stronger noise std=0.2
        noisy_images = images.clone()
        noisy_images += torch.randn_like(images) * std + mean
        return noisy_images

    noisy_images_train = add_gaussian_noise(clean_images_train)
    noisy_images_test = add_gaussian_noise(clean_images_test)

    # Organize the images into dictionaries
    x_in_train = {'HR': clean_images_train, 'SR': noisy_images_train}
    x_in_test = {'HR': clean_images_test, 'SR': noisy_images_test}

    x_in_train_original = {'HR': clean_images_train, 'SR': noisy_images_train}

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

    #  Visualize a tensor
    def visualize_tensor(image_tensor, title=None):
        # Ensure the tensor has the correct shape [3, height, width]
        if len(image_tensor.shape) != 3 or image_tensor.shape[0] != 3:
            print("Error: Invalid tensor shape. Expected [3, height, width].")
            return

        # Convert the tensor to a numpy array and transpose dimensions for visualization
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Display the image using matplotlib
        plt.imshow(image_np)
        plt.axis('off')  # Hide axes
        if title:
            plt.title(title)
        plt.show()


    # Visualize the images
    # visualize_images(x_in_train['HR'][0], x_in_train['SR'][0])

    print(x_in_train['HR'][0].shape)
    # visualize_tensor(x_in_train['HR'][0],'Original.. before everything')

    print('Status: Data Loaded Successfully')

# *********************************************
# Step 2: inference

print('Status: Inference Time...')

# Load a trained denoiser...
denoise_fun = UNet(    image_size=32
)
denoise_fun.load_state_dict(torch.load('denoising_simple.pth'))
denoise_fun.eval()

config_diff = {
    'beta_start': 0.0001,
    'beta_end': 0.5,
    'num_steps': 50,      # Reduced number of steps
    'schedule': "quad"
}

diffusion = GaussianDiffusion(denoise_fun, image_size=(32,32),channels=3,loss_type='l1',conditional=True,config_diff=config_diff)
diffusion.load_state_dict(torch.load('diffusion_model.pth'))

print('Status: Diffusion and denoising model loaded sucesfully')


# Inference
print(len(x_in_test['SR']))
print(x_in_test['SR'][0].shape)

visualize_tensor(x_in_train_original['HR'][3], 'Original HighQuality')
visualize_tensor(x_in_train_original['SR'][3], 'Original CrapQuality')
visualize_tensor(diffusion.p_sample_loop_single(x_in_train['SR'][3]), 'Sampled Image')