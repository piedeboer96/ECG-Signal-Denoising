import os
import scipy.io
import matplotlib.pyplot as plt

# Directory containing the .mat files
directory = 'noisy_samples/samples'

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
af_mat_HR = scipy.io.loadmat('noisy_samples/af_sig_HR.mat')
af_sig_HR = af_mat_HR['sig_HR'].squeeze()

# ARDB Signal Clean 
ardb_mat_HR = scipy.io.loadmat('noisy_samples/ardb_sig_HR.mat')
ardb_sig_HR = ardb_mat_HR['sig_HR'].squeeze()

# Clean Signals
signals_HR = [af_sig_HR] * 4 + [ardb_sig_HR] * 8

# Noisy Signals (for inferece)
signals_SR = signals_in_order



#####################
visualize_data = 1

if visualize_data == 1:

    # Plot Clean Signals
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(signals_HR, 1):
        plt.subplot(4, 4, i)
        plt.plot(signal)
        plt.title(f'Clean Signal {i}', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Plot Noisy Signals
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(signals_SR, 1):
        plt.subplot(4, 4, i)
        plt.plot(signal)
        plt.title(f'Noisy Signal {i}', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()