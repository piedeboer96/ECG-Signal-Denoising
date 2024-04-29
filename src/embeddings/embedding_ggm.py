import numpy as np
import torch
import pickle
import tqdm
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
    def ggm_to_ecg(ggm):

        # Undo the permuation
        ggm_restored = torch.permute(ggm, (1, 2, 0))

        # Convetr to numpy
        ggm = ggm_restored.detach().cpu().numpy()

        # Extract GASF channel
        gasf_channel = ggm[:, :, 0]
        
        # Extract diagonal elements
        diagonals = np.diagonal(gasf_channel)

        return diagonals

    def build_ggm_data(clean_slices_pkl, noisy_slice_pkl, k=1):

        embedding_GGM = EmbeddingGGM()

        with open(clean_slices_pkl, 'rb') as f:
            clean_slices = pickle.load(f)

        with open(noisy_slice_pkl, 'rb') as f:
            noisy_slices = pickle.load(f)

        # Display Info
        num_of_slices = len(clean_slices); print('Number of slices',num_of_slices)
        len_of_slice = len(noisy_slices);  print('Length of slice', len_of_slice)

        # SMALL
        num_of_slices_small = int(num_of_slices/k)
        print('Number of slices',num_of_slices_small)

        # Spectrograms
        ggms_clean = []
        ggms_noisy = []

        # Parameters
        sampto=512

        # Build GGM Embeddings
        for i in tqdm(range(num_of_slices_small)):  # NOTE : small 

            #######
            ecg_clean =  torch.tensor(clean_slices[i][:sampto])
            ecg_noisy_EM = torch.tensor(noisy_slices[i][:sampto])
            
            #######
            ggm_clean = embedding_GGM.ecg_to_ggm(ecg_clean)
            ggm_noisy = embedding_GGM.ecg_to_ggm(ecg_noisy_EM)

            #######
            ggms_clean.append(ggm_clean)
            ggms_noisy.append(ggm_noisy)

        ggms_clean = ggm_clean
        ggms_noisy = ggm_noisy

        # Define a custom PyTorch dataset 
        class GGMdataset(Dataset):
            def __init__(self, ggms_clean, ggms_noisy, transform=None):
                self.ggms_clean = ggms_clean
                self.ggms_noisy = ggms_noisy
                self.transform = transform

            def __len__(self):
                return len(self.ggms_clean)

            def __getitem__(self, idx):
                spec_clean = self.ggms_clean[idx]
                spec_noisy = self.ggms_noisy[idx]

                if self.transform:
                    spec_clean = self.transform(spec_clean)
                    spec_noisy = self.transform(spec_noisy)

                return spec_clean, spec_noisy

        # Split the data into training and validation sets
        ggms_clean_train, ggms_clean_val, ggms_noisy_train, ggms_noisy_val = train_test_split(
            ggms_clean, ggms_noisy, test_size=0.2, random_state=42)

        # Create datasets for training and validation
        train_dataset = GGMdataset(ggms_clean_train, ggms_noisy_train)
        val_dataset = GGMdataset(ggms_clean_val, ggms_noisy_val)

        # Adapt to SR3 format
        x_in_train = {'HR': ggms_clean_train, 'SR': ggms_noisy_train}
        x_in_test = {'HR': ggms_clean_val, 'SR': ggms_noisy_val}

        return x_in_train, x_in_test