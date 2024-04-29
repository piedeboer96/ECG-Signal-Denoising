import numpy as np
import torch
# import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm  # Corrected import statement
from pyts.image import MarkovTransitionField
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class EmbeddingGGM:
    def __init__(self):
        # Load the Markov Transition Field object
        self.mtf = MarkovTransitionField()

    def rescale_time_series(self, X):
        """
        Rescale the time series X to fall within the interval [-1, 1].

        Parameters:
        X (array-like): The input time series.

        Returns:
        X_rescaled (array-like): The rescaled time series.
        """
        min_ = np.amin(X)
        max_ = np.amax(X)
        scaled_serie = (2*X - max_ - min_)/(max_ - min_)
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        return scaled_serie

    def ecg_to_ggm(self, X):
        X = self.rescale_time_series(X)

        r = np.arange(1, len(X) + 1) / len(X)
        phi = np.arccos(X)

        N = len(X)
        GASF = np.zeros((N, N))
        GADF = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                GASF[i, j] = np.cos((phi[i] + phi[j]) / 2)
                GADF[i, j] = np.cos((phi[i] - phi[j]) / 2)

        MTF = self.mtf.transform(X.reshape(1, -1))[0]

        ggm = np.stack((GASF, GADF, MTF), axis=-1)
        x = torch.tensor(ggm)
        x = torch.permute(x, (2, 0, 1))

        return x  

    # Reconstruct
    def ggm_to_ecg(self, ggm):

        # Undo the permuation
        ggm_restored = torch.permute(ggm, (1, 2, 0))

        # Convetr to numpy
        ggm = ggm_restored.detach().cpu().numpy()

        # Extract GASF channel
        gasf_channel = ggm[:, :, 0]
        
        # Extract diagonal elements
        diagonals = np.diagonal(gasf_channel)

        return diagonals

    def visualize_tensor(self,tensor):
        print('Shape', tensor.shape)

        # Convert the tensor to a NumPy array
        image_array = tensor.numpy()

        # Transpose the array to (H, W, C) format
        image_array = image_array.transpose(1, 2, 0)

        # Display the image using Matplotlib
        plt.imshow(image_array)
        plt.axis('off')  # Turn off axis
        plt.show()

    def build_ggm_data(self, clean_slices, noisy_slices, k):

        embedding_GGM = EmbeddingGGM()

        # Display Info
        num_of_slices = len(clean_slices); print('Number of slices',num_of_slices)
        len_of_slice = len(noisy_slices);  print('Length of slice', len_of_slice)

        # SMALL
        print('k', k)
        num_of_slices_small = int(num_of_slices/k)
        print('Number of slices',num_of_slices_small)

        # ggmtrograms
        ggms_clean = []
        ggms_noisy = []

        # Parameters
        sampto=512

        # Build GGM Embeddings
        for i in tqdm(range(num_of_slices_small)):  # NOTE : small 

            #######
            ecg_clean =  clean_slices[i][:sampto]
            ecg_noisy_EM = noisy_slices[i][:sampto]
            
            #######
            ggm_clean = embedding_GGM.ecg_to_ggm(ecg_clean)
            ggm_noisy = embedding_GGM.ecg_to_ggm(ecg_noisy_EM)

            #######
            ggms_clean.append(ggm_clean)
            ggms_noisy.append(ggm_noisy)

        # ggms_clean = ggm_clean
        # ggms_noisy = ggm_noisy

        ggms_clean = [tensor.float() for tensor in ggms_clean]  # float.64 --> float.32
        ggms_noisy = [tensor.float() for tensor in ggms_noisy]

        # Define a custom PyTorch dataset 
        class GGMdataset(Dataset):
            def __init__(self, ggms_clean, ggms_noisy, transform=None):
                self.ggms_clean = ggms_clean
                self.ggms_noisy = ggms_noisy
                self.transform = transform

            def __len__(self):
                return len(self.ggms_clean)

            def __getitem__(self, idx):
                ggm_clean = self.ggms_clean[idx]
                ggm_noisy = self.ggms_noisy[idx]

                if self.transform:
                    ggm_clean = self.transform(ggm_clean)
                    ggm_noisy = self.transform(ggm_noisy)

                return ggm_clean, ggm_noisy

        # Split the data into training and validation sets
        ggms_clean_train, ggms_clean_val, ggms_noisy_train, ggms_noisy_val = train_test_split(
            ggms_clean, ggms_noisy, test_size=0.2, random_state=42)

        # Create datasets for training and validation
        train_dataset = GGMdataset(ggms_clean_train, ggms_noisy_train)
        val_dataset = GGMdataset(ggms_clean_val, ggms_noisy_val)

        # Adapt to SR3 format
        x_in_train = {'HR': ggms_clean_train, 'SR': ggms_noisy_train}
        x_in_test = {'HR': ggms_clean_val, 'SR': ggms_noisy_val}

        # print(x_in_train['HR'][0].shape)
        embedding_GGM.visualize_tensor(x_in_train['HR'][0])


        return x_in_train, x_in_test