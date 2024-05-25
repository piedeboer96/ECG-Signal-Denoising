import os
import scipy.io
import matplotlib.pyplot as plt



class DataHelper:
    def __init__(self) -> None:
        pass

    def load_data_from_directory(self, path_to_noisy_samples, path_clean_AF, path_clean_ARDB):

        # Directory containing the .mat files
        directory = path_to_noisy_samples

        # Get a sorted list of filenames in the directory
        filenames = sorted(os.listdir(directory))

        signals_in_order = []

        signal_names = []

        # Iterate over the sorted filenames in alphabetical order
        for filename in filenames:
            if filename.endswith('.mat'):

                print(filename)
                # Load the .mat file
                data = scipy.io.loadmat(os.path.join(directory, filename))
                
                # Extract the signal
                sig = data[filename[:-4]].squeeze()
                
                signals_in_order.append(sig)
                signal_names.append(filename[:-4])


        ################################################

        # AF Signal Clean
        af_mat_HR = scipy.io.loadmat(path_clean_AF)
        af_sig_HR = af_mat_HR['sig_HR'].squeeze()

        # ARDB Signal Clean 
        ardb_mat_HR = scipy.io.loadmat(path_clean_ARDB)
        ardb_sig_HR = ardb_mat_HR['sig_HR'].squeeze()

        # Clean Signals
        signals_HR = [af_sig_HR] * 4 + [ardb_sig_HR] * 12

        # Noisy Signals (for inferece)
        signals_SR = signals_in_order

        return signals_HR, signals_SR


