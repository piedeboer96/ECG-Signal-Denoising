import matplotlib.pyplot as plt
import torch
import pickle
import scipy.io
from torch import device
from datetime import datetime

from diffusion import GaussianDiffusion
from unet import UNet
from embedding import EmbeddingGAF

from visualizations import Visualizations
from noisy_ecg_builder import NoisyECGBuilder

# ! NOTE: INFERENCE NEEDS TO BE ON CPU , fix this!
device = 'cpu'

# Define parameters of the U-Net (denoising function)
in_channels = 1*2                        
out_channels = 1                        # Output will also be GrayScale
inner_channels = 32                     # Depth feature maps, model complexity 
norm_groups = 32                        # Granularity of normalization, impacting convergence
channel_mults = (1, 2, 4, 8, 8)
attn_res = [8]
res_blocks = 3
dropout = 0
with_noise_level_emb = True
image_size = 128

# Load a trained denoiser...
denoise_fun = UNet(
    in_channel=in_channels,
    out_channel=out_channels,
    inner_channel=inner_channels,
    norm_groups=norm_groups,
    channel_mults=channel_mults,
    attn_res=attn_res,
    res_blocks=res_blocks,
    dropout=dropout,
    with_noise_level_emb=with_noise_level_emb,
    image_size=128
).to(device)  # Move the denoising model to the GPU if available


# Noise schedule from SR3 Github
config_diff = {
    'beta_start': 1e-6,
    'beta_end': 1e-2,
    'num_steps': 2000,      # Reduced number of steps
    'schedule': "linear"
}

#################################
denoise_fun.load_state_dict(torch.load('src/generative_models/dn_model_MA17h00.pth', map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load('src/generative_models/diff_model_MA17h00.pth', map_location=device))

print('Status: Diffusion and denoising model loaded successfully')
    
#################################
embedding_gaf = EmbeddingGAF()
nb = NoisyECGBuilder()

# LOAD DATA
with open('src/generative_models/ardb_slices_clean_MA.pkl', 'rb') as f:
    clean_signals = pickle.load(f)

# sig_HR = clean_signals[52222][:128]       # IN TRAINING SET
sig_HR = clean_signals[57200][:128]

gaf_HR = embedding_gaf.ecg_to_GAF(sig_HR)

del clean_signals                           # REMOVE FROM MEMORY

with open('src/generative_models/ardb_slices_noisy_MA_snr3.pkl', 'rb') as f:
    noisy_signals = pickle.load(f)

# sig_SR = noisy_signals[52222][:128]

sig_SR = nb.add_noise_to_ecg(sig_HR, noise_type='ma',snr=3)     # IT WAS VERY LIKELY NOT IN THE TRAINING SET -- SINCE OUR EM NOISE... IS NEW RANDOM PICKED :)

gaf_SR = embedding_gaf.ecg_to_GAF(sig_SR)

del noisy_signals                           # REMOVE FROM MEMORY 

############################
# INFERENCE (NOT IN TRAINING SET)

# FLOAT.32
x = gaf_SR.to("cpu")   
x = x.to(torch.float32)

# SAMPLE TENSOR
sampled_tensor = diffusion.p_sample_loop_single(x)
sampled_tensor = sampled_tensor.unsqueeze(0)

# SAVE 
hour, minute = datetime.now().hour, datetime.now().minute
formatted_time = f"{hour}h{minute:02d}"
save_tensor_sample = 'gaf_sampled_' + str(formatted_time) + '.pkl'
with open(save_tensor_sample,'wb') as f:
    pickle.dump(sampled_tensor, f)

# RECOVER
sig_rec = embedding_gaf.GAF_to_ecg(sampled_tensor)
filename = 'sig_rec.mat'

# Save the array to a .mat file
scipy.io.savemat(filename, {'sig_rec': sig_rec})

#####################
#####################

vis = Visualizations()
vis.visualize_tensor(sampled_tensor)
vis.plot_multiple_timeseries([sig_HR, sig_SR, sig_rec], ['HR', 'SR', 'Recovered'])

#####################
#####################


