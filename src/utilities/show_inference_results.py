import torch
import matplotlib.pyplot as plt

# Load the saved inference results
inf_result = torch.load('inf_single_result.pth')



print(inf_result.shape)
print(len(inf_result))
print(type(inf_result))

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
visualize_results(inf_result)
