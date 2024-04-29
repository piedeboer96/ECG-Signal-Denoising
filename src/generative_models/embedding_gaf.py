import numpy as np
import torch
# import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm  # Corrected import statement
from pyts.image import MarkovTransitionField
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Further Reading: 
#       https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3
class EmbeddingGAF:
    def __init__(self):
        pass

    def rescale_time_series(self, X, VMIN=-1, VMAX=1):
        """
        Rescale the time series X to fall within the interval [-1, 1].

        Parameters:
        X (array-like): The input time series.
        VMIN (float): Minimum value for rescaling.
        VMAX (float): Maximum value for rescaling.

        Returns:
        X_rescaled (array-like): The rescaled time series.
        """
        # Min-Max scaling:
        min_ = np.amin(X)
        max_ = np.amax(X)
        scaled_serie = (2 * X - max_ - min_) / (max_ - min_)
        
        # Use np.core.umath.maximum and np.core.umath.minimum for faster rescaling
        X_rescaled = np.core.umath.maximum(np.core.umath.minimum(scaled_serie, VMAX), VMIN)

        return X_rescaled

    def ecg_to_GAF(self, X):

        # Rescale
        X = self.rescale_time_series(X)

        # Calculate the angular values 'phi' using the rescaled time series
        phi = np.arccos(X)

        # Compute GASF matrix
        N = len(X)
        phi_matrix = np.tile(phi, (N, 1))
        GASF = np.cos((phi_matrix + phi_matrix.T) / 2)

        # Return Tensor (1,x,x) for GASF
        x = torch.tensor(GASF).unsqueeze(0)

        return x

    # Reconstruct
    def GAF_to_ecg(self, gaf):

        # print(gaf.shape)
        restored_ecg = gaf.detach().cpu().numpy()
        
        diagonals = np.diagonal(restored_ecg)

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

    def build_gaf_data(self, clean_slices, noisy_slices, k):

        num_of_slices = len(clean_slices)
        
        print('k', k)
        num_of_slices_small = int(num_of_slices/k)
        print('Number of slices',num_of_slices_small)

        # gaftrograms
        gafs_clean = []
        gafs_noisy = []

        # Parameters
        sampto=512

        # Build gaf Embeddings
        for i in tqdm(range(num_of_slices_small)):  # NOTE : small 

            #######
            ecg_clean =  clean_slices[i][:sampto]
            ecg_noisy_EM = noisy_slices[i][:sampto]
            
            #######
            gaf_clean = self.ecg_to_GAF(ecg_clean)
            gaf_noisy = self.ecg_to_GAF(ecg_noisy_EM)

            #######
            gafs_clean.append(gaf_clean)
            gafs_noisy.append(gaf_noisy)
        

        # float.32 for SR3 model
        gafs_clean = [tensor.float() for tensor in gafs_clean]  # float.64 --> float.32
        gafs_noisy = [tensor.float() for tensor in gafs_noisy]

        # Define a custom PyTorch dataset 
        class gafdataset(Dataset):
            def __init__(self, gafs_clean, gafs_noisy, transform=None):
                self.gafs_clean = gafs_clean
                self.gafs_noisy = gafs_noisy
                self.transform = transform

            def __len__(self):
                return len(self.gafs_clean)

            def __getitem__(self, idx):
                gaf_clean = self.gafs_clean[idx]
                gaf_noisy = self.gafs_noisy[idx]

                if self.transform:
                    gaf_clean = self.transform(gaf_clean)
                    gaf_noisy = self.transform(gaf_noisy)

                return gaf_clean, gaf_noisy

        # Split the data into training and validation sets
        gafs_clean_train, gafs_clean_val, gafs_noisy_train, gafs_noisy_val = train_test_split(
            gafs_clean, gafs_noisy, test_size=0.2, random_state=42)

        # Create datasets for training and validation
        train_dataset = gafdataset(gafs_clean_train, gafs_noisy_train)
        val_dataset = gafdataset(gafs_clean_val, gafs_noisy_val)

        # Adapt to SR3 format
        x_in_train = {'HR': gafs_clean_train, 'SR': gafs_noisy_train}
        x_in_test = {'HR': gafs_clean_val, 'SR': gafs_noisy_val}

        return x_in_train, x_in_test
