import torch
import torchaudio
import pickle
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EmbeddingSpec:
    def ecg_to_spectrogram_log(self, ecg_signal, n_fft, hop_length):
        """
        Convert ECG signal to spectrogram using torchaudio library.

        Args:
        - ecg_signal (torch.Tensor): 1D tensor representing the ECG signal.
        - n_fft (int): Size of FFT window. Default is 400.
        - hop_length (int or None): Number of samples between successive frames.
          If None, defaults to n_fft / 4. Default is None.

        Returns:
        - torch.Tensor: Spectrogram of the ECG signal.
        """
        # Reshape ECG signal to (batch_size, num_channels, signal_length)
        ecg_signal = ecg_signal.unsqueeze(0).unsqueeze(0)

        # Compute spectrogram
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(ecg_signal)

        # Take the log
        spec = spectrogram[0]
        spec_log = torch.log10(spec)

        return spec_log

    def spectrogram_log_to_ecg(self, spec_log, n_fft, hop_length, n_iter=400):
        """
        Convert spectrogram back to ECG signal using torchaudio library.

        Args:
        - spectrogram (torch.Tensor): Spectrogram of the ECG signal.
        - n_fft (int): Size of FFT window. Default is 400.
        - hop_length (int or None): Number of samples between successive frames.
          If None, defaults to n_fft / 4. Default is None.

        Returns:
        - torch.Tensor: Reconstructed ECG signal.
        """
        # Undo log operation
        spec = torch.pow(10, spec_log)

        # Compute inverse spectrogram
        ecg_signal = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=n_iter, hop_length=hop_length)(spec)

        # Squeeze the batch and channel dimensions
        ecg_signal = ecg_signal.squeeze(0).squeeze(0)

        return ecg_signal
    
    ## NOTE: 
    ## >> this one needs improvement...
    def sampled_spec_to_ecg(sampled_spec, min_value, max_value, n_fft, hop_length, n_iter):
        
        embedding_spec = EmbeddingSpec()
        # Undo normalization
        spec_log = embedding_spec.undo_normalization_tensor(sampled_spec, min_value, max_value)

        # Undo log operation
        spec = torch.pow(10, spec_log)

        # Compute inverse spectrogram
        ecg_signal = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=n_iter, hop_length=hop_length)(spec)

        # Squeeze the batch and channel dimensions
        ecg_signal = ecg_signal.squeeze(0).squeeze(0)

        return ecg_signal

    def build_spec_data(clean_slices_pkl, noisy_slice_pkl, k=1):

        embedding_spec = EmbeddingSpec()

        with open(clean_slices_pkl, 'rb') as f:
            clean_slices = pickle.load(f)

        with open(noisy_slice_pkl, 'rb') as f:
            noisy_slices = pickle.load(f)

        # Display Info
        num_of_slices = len(clean_slices);   print('Number of slices',num_of_slices)
        len_of_slice = len(noisy_slices); print('Length of slice', len_of_slice)

        # SMALL
        num_of_slices_small = int(num_of_slices/k)
        print('Number of slices',num_of_slices_small)

        # Spectrograms
        specs_clean = []
        specs_noisy_EM = []

        # # Spectrogram Parameters
        n_fft = 126
        hop_length = 8
        sampto = (62*8) + 8 

        # Build Spectrograms
        for i in tqdm(range(num_of_slices_small)):  # NOTE : small 

            #######
            ecg_clean =  torch.tensor(clean_slices[i][:sampto])
            ecg_noisy_EM = torch.tensor(noisy_slices[i][:sampto])
            
            #######
            spec_clean = embedding_spec.ecg_to_spectrogram_log(ecg_clean, n_fft, hop_length)
            spec_clean = embedding_spec.normalize_tensor(spec_clean)                                    # normalize
            spec_noisy_EM = embedding_spec.ecg_to_spectrogram_log(ecg_noisy_EM, n_fft, hop_length)
            spec_noisy_EM = embedding_spec.normalize_tensor(spec_noisy_EM)                              # normalize

            #######
            specs_clean.append(spec_clean)
            specs_noisy_EM.append(spec_noisy_EM)

        # specs_clean = specs_cleanls
        # specs_noisy = specs_noisy_EM

        specs_clean = [tensor.float() for tensor in specs_clean]  # float.64 --> float.32
        specs_noisy = [tensor.float() for tensor in specs_noisy_EM]

        # Define a custom PyTorch dataset 
        class SpectrogramDataset(Dataset):
            def __init__(self, specs_clean, specs_noisy, transform=None):
                self.specs_clean = specs_clean
                self.specs_noisy = specs_noisy
                self.transform = transform

            def __len__(self):
                return len(self.specs_clean)

            def __getitem__(self, idx):
                spec_clean = self.specs_clean[idx]
                spec_noisy = self.specs_noisy[idx]

                if self.transform:
                    spec_clean = self.transform(spec_clean)
                    spec_noisy = self.transform(spec_noisy)

                return spec_clean, spec_noisy

        # Split the data into training and validation sets
        specs_clean_train, specs_clean_val, specs_noisy_train, specs_noisy_val = train_test_split(
            specs_clean, specs_noisy, test_size=0.2, random_state=42)

        # Create datasets for training and validation
        train_dataset = SpectrogramDataset(specs_clean_train, specs_noisy_train)
        val_dataset = SpectrogramDataset(specs_clean_val, specs_noisy_val)

        # Adapt to SR3 format
        x_in_train = {'HR': specs_clean_train, 'SR': specs_noisy_train}
        x_in_test = {'HR': specs_clean_val, 'SR': specs_noisy_val}

        return x_in_train, x_in_test
    
    def normalize_tensor(input_tensor):
        min_val = torch.min(input_tensor)
        max_val = torch.max(input_tensor)

        # Normalize the tensor between -1 and 1
        normalized_tensor = 2 * ((input_tensor - min_val) / (max_val - min_val)) - 1

        return normalized_tensor
    
    def undo_normalization_tensor(normalized_tensor, min_val, max_val):     # NOTE: when reconstruction, use min_val and max_val from distrubution

        # Undo normalization
        original_tensor = ((normalized_tensor + 1) / 2) * (max_val - min_val) + min_val

        return original_tensor

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