import scipy.io
import numpy as np
import pickle
import matplotlib.pyplot as plt

from noisy_ecg_builder import NoisyECGBuilder

#####################

nb = NoisyECGBuilder()

########################
# Electrode Motion

# Clean Signal
data_HR = 'noisy_samples/ardb/ardb_sig_HR'
mat_HR = scipy.io.loadmat(data_HR)
sig_HR = mat_HR['sig_HR'].squeeze()

# Electrode Motion
data_EM = 'noisy_samples/slices/em_slice_ind.mat'
mat_EM = scipy.io.loadmat(data_EM)
sig_EM = mat_EM['em_new'].squeeze()

# Add Noise at SNR {0,5,10,15}
ardb_sig_SR_em_snr_00 = nb.noise_adder(sig_HR, sig_EM, 0)
ardb_sig_SR_em_snr_05 = nb.noise_adder(sig_HR, sig_EM, 5)
ardb_sig_SR_em_snr_10 = nb.noise_adder(sig_HR, sig_EM, 10)
ardb_sig_SR_em_snr_15 = nb.noise_adder(sig_HR, sig_EM, 15)

########################
# Muscle Artifact

# Muscle Artifact Noise
data_MA = 'noisy_samples/slices/ma_slice_ind.mat'
mat_MA = scipy.io.loadmat(data_MA)
sig_MA = mat_MA['ma_slice'].squeeze()

# Add Nosie at SNR {0,5,10,15}
ardb_sig_SR_ma_snr_00 = nb.noise_adder(sig_HR, sig_MA, 0)
ardb_sig_SR_ma_snr_05 = nb.noise_adder(sig_HR, sig_MA, 5)
ardb_sig_SR_ma_snr_10 = nb.noise_adder(sig_HR, sig_MA, 10)
ardb_sig_SR_ma_snr_15 = nb.noise_adder(sig_HR, sig_MA, 15)

########################
# Composite Noise

# Raw Signal
with open('noisy_samples/slices/ardb_raw_slices.pkl', 'rb') as file:
    sig_raw = pickle.load(file)[57200][:128]

data_RAW = 'noisy_samples/af/af_sig_raw.mat'
mat_RAW = scipy.io.loadmat(data_RAW)
sig_RAW_af = mat_RAW['sig_HR']     

# Composite Noise
data_COMP = 'noisy_samples/slices/comp_slice_ind.mat'
mat_COMP = scipy.io.loadmat(data_COMP)
sig_COMP = mat_COMP['comp_slice_ind'].squeeze()

# Add noise at SNR {0,5,10,15}
ardb_sig_SR_comp_snr_00 = nb.noise_adder(sig_raw, sig_COMP, 0)
ardb_sig_SR_comp_snr_05 = nb.noise_adder(sig_raw, sig_COMP, 5)
ardb_sig_SR_comp_snr_10 = nb.noise_adder(sig_raw, sig_COMP, 10)
ardb_sig_SR_comp_snr_15 = nb.noise_adder(sig_raw, sig_COMP, 15)

# Add noise at SNR 
af_sig_SR_comp_snr_00 = nb.noise_adder(sig_RAW_af, sig_COMP, 0)
af_sig_SR_comp_snr_05 = nb.noise_adder(sig_RAW_af, sig_COMP, 5)
af_sig_SR_comp_snr_10 = nb.noise_adder(sig_RAW_af, sig_COMP, 10)
af_sig_SR_comp_snr_15 = nb.noise_adder(sig_RAW_af, sig_COMP, 15)

# Save ardb_sig_SR_em
scipy.io.savemat('ardb_sig_SR_em_snr_00.mat', {'ardb_sig_SR_em_snr_00': ardb_sig_SR_em_snr_00})
scipy.io.savemat('ardb_sig_SR_em_snr_05.mat', {'ardb_sig_SR_em_snr_05': ardb_sig_SR_em_snr_05})
scipy.io.savemat('ardb_sig_SR_em_snr_10.mat', {'ardb_sig_SR_em_snr_10': ardb_sig_SR_em_snr_10})
scipy.io.savemat('ardb_sig_SR_em_snr_15.mat', {'ardb_sig_SR_em_snr_15': ardb_sig_SR_em_snr_15})

# Save ardb_sig_SR_ma
scipy.io.savemat('ardb_sig_SR_ma_snr_00.mat', {'ardb_sig_SR_ma_snr_00': ardb_sig_SR_ma_snr_00})
scipy.io.savemat('ardb_sig_SR_ma_snr_05.mat', {'ardb_sig_SR_ma_snr_05': ardb_sig_SR_ma_snr_05})
scipy.io.savemat('ardb_sig_SR_ma_snr_10.mat', {'ardb_sig_SR_ma_snr_10': ardb_sig_SR_ma_snr_10})
scipy.io.savemat('ardb_sig_SR_ma_snr_15.mat', {'ardb_sig_SR_ma_snr_15': ardb_sig_SR_ma_snr_15})

# Save ardb_sig_SR_comp
scipy.io.savemat('ardb_sig_SR_comp_snr_00.mat', {'ardb_sig_SR_comp_snr_00': ardb_sig_SR_comp_snr_00})
scipy.io.savemat('ardb_sig_SR_comp_snr_05.mat', {'ardb_sig_SR_comp_snr_05': ardb_sig_SR_comp_snr_05})
scipy.io.savemat('ardb_sig_SR_comp_snr_10.mat', {'ardb_sig_SR_comp_snr_10': ardb_sig_SR_comp_snr_10})
scipy.io.savemat('ardb_sig_SR_comp_snr_15.mat', {'ardb_sig_SR_comp_snr_15': ardb_sig_SR_comp_snr_15})

# Save af_sig_SR_comp
scipy.io.savemat('af_sig_SR_comp_snr_00.mat', {'af_sig_SR_comp_snr_00': af_sig_SR_comp_snr_00})
scipy.io.savemat('af_sig_SR_comp_snr_05.mat', {'af_sig_SR_comp_snr_05': af_sig_SR_comp_snr_05})
scipy.io.savemat('af_sig_SR_comp_snr_10.mat', {'af_sig_SR_comp_snr_10': af_sig_SR_comp_snr_10})
scipy.io.savemat('af_sig_SR_comp_snr_15.mat', {'af_sig_SR_comp_snr_15': af_sig_SR_comp_snr_15})