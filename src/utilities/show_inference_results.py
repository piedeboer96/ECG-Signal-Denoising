import torch
import matplotlib.pyplot as plt

# Load the saved inference results
inference_results = torch.load('inference_results.pth')



print(inference_results.shape)
print(len(inference_results))
print(type(inference_results))

# Visualize the inference results
def visualize_results(results):
    num_images = results.size(0)  # Get the number of images
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(results[i])  # Assuming image tensor is in HWC format
        ax.axis('off')

    plt.show()

# Call the visualization function
visualize_results(inference_results)
