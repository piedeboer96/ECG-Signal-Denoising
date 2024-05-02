import matplotlib.pyplot as plt
import numpy as np
import pickle

from embedding import EmbeddingGAF

def visualize_tensor(tensor):
    print('Shape', tensor.shape)

    # Convert the tensor to a NumPy array
    image_array = tensor.numpy()

    # Transpose the array to (H, W, C) format
    image_array = image_array.transpose(1, 2, 0)

    # Display the image using Matplotlib
    plt.imshow(image_array)
    plt.axis('off')  # Turn off axis
    plt.show()

with open('ggm_sampled_16h30', 'rb') as f:
    tensor_data = pickle.load(f)

tensor_data = tensor_data.squeeze(0)
visualize_tensor(tensor_data)


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

embedding_gaf = EmbeddingGAF()

time_series = embedding_gaf.GAF_to_ecg(tensor_data)

print(time_series.shape)

plot_multiple_timeseries([time_series], [' Recon'])
