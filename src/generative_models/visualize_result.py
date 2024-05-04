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

with open('src\generative_models\ggm_sampled_13h39.pkl', 'rb') as f:
    tensor_data_ggm = pickle.load(f)

with open('src\generative_models\gaf_sampled_14h48.pkl', 'rb') as f:
    tensor_data_gaf = pickle.load(f)

tensor_data_ggm = tensor_data_ggm.squeeze(0)
visualize_tensor(tensor_data_ggm)

tensor_data_gaf = tensor_data_gaf
visualize_tensor(tensor_data_gaf)

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

ts_ggm = embedding_gaf.GAF_to_ecg(tensor_data_ggm)
ts_gaf = embedding_gaf.GAF_to_ecg(tensor_data_gaf)

print(ts_gaf.shape)

plot_multiple_timeseries([ts_gaf], [' Recon'])
