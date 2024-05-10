import pickle

from noisy_ecg_builder import NoisyECGBuilder
from vis import Visualizations

# TODO:
#   - generate noise at different SNRs

nb = NoisyECGBuilder()
vis = Visualizations()

# ***********************************
# LOAD DATA
with open('src/generative_models/ardb_slices_clean.pkl', 'rb') as f:
    clean_signals = pickle.load(f)                                              ### CLEAN SIGNAL

sig_HR = clean_signals[52222][:128]

del clean_signals                           # REMOVE FROM MEMORY

with open('src/generative_models/ardb_slices_noisy.pkl', 'rb') as f:
    noisy_signals = pickle.load(f)

sig_SR = noisy_signals[52222][:128]

sig_SR_build = nb.add_noise_to_ecg(sig_HR,'ma',snr=3)

# gaf_SR = embedding_gaf.ecg_to_GAF(sig_SR)

del noisy_signals                           # REMOVE FROM MEMORY

vis.plot_multiple_timeseries([sig_HR, sig_SR, sig_SR_build], ['HR', 'SR', 'SR Build'])

# scipy.io.savemat('sig_HR.mat', {'sig_HR': sig_HR})
# scipy.io.savemat('sig_SR.mat', {'sig_SR': sig_SR})

# ***********************************
# GENERATE DATA TO COMPARE THE MODELS 




