import matplotlib.pyplot as plt
import numpy as np
import pickle

from embedding_gaf import EmbeddingGAF

# tensor_pickle_path = 'src/generative_models/sampled_tensor_gaf_cuda_9h35.pkl'

# with open('src/generative_models/clean_slices_270_samples.pkl', 'rb') as f:
#     clean_slices = pickle.load(f)

# with open('/Users/piedeboer/Desktop/Thesis/code/signal-denoising/src/generative_models/noisy_sllices_270_samples.pkl', 'rb') as f:
#     noisy_slices = pickle.load(f)

# # clean_slices[0]
# # noisy_slices[4]

# embedding_gaf = EmbeddingGAF()

# gaf_SR = embedding_gaf.ecg_to_GAF(clean_slices[0])
# gaf_HR = embedding_gaf.ecg_to_GAF(noisy_slices[0])

# # Load the tensor from the pickle file
# with open(tensor_pickle_path, 'rb') as f:
#     tensor_data = pickle.load(f)

# # # Load the tensor from the pickle file
# # with open(tensor_pickle_path, 'rb') as f:
# #     tensor_data = pickle.load(f)

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

with open('gaf_sampled_12h24', 'rb') as f:
    tensor_data = pickle.load(f)


# visualize_tensor(gaf_HR)
# visualize_tensor(gaf_SR)
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

plot_multiple_timeseries([time_series], [' Recon'])
