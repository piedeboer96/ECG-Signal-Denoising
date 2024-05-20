import torch
import scipy.io
from torch import device

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
denoise_fun.load_state_dict(torch.load('models/dn_model_COMP_AF17h25.pth', map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load('models/diff_model_COMP_AF17h25.pth', map_location=device))

print('Status: Diffusion and denoising model loaded successfully')
    
#################################
vis = Visualizations()
embedding_gaf = EmbeddingGAF()
nb = NoisyECGBuilder()
#################################

data_HR = 'results/ardb/MA/m2_ma_snr_5/sig_HR.mat'
data_SR = 'results/ardb/MA/m2_ma_snr_5/sig_SR.mat'

#Load sig_HR from .mat file
mat_HR = scipy.io.loadmat(data_HR)
sig_HR = mat_HR['sig_HR'].squeeze()

mat_SR = scipy.io.loadmat(data_SR)
sig_SR = mat_SR['sig_SR'].squeeze()

gaf_HR = embedding_gaf.ecg_to_GAF(sig_HR)
gaf_SR = embedding_gaf.ecg_to_GAF(sig_SR)

vis.plot_multiple_timeseries([sig_HR,sig_SR],['Original', 'Noisy'])

############################
# INFERENCE (NOT IN TRAINING SET)

# FLOAT.32
x = gaf_SR.to("cpu")   
x = x.to(torch.float32)

# SAMPLE TENSOR
sampled_tensor = diffusion.p_sample_loop_single(x)
sampled_tensor = sampled_tensor.unsqueeze(0)


# RECOVER
sig_rec = embedding_gaf.GAF_to_ecg(sampled_tensor)

filename_SR = 'sig_SR.mat'
filename_HR = 'sig_HR.mat'
filename_rec = 'sig_rec.mat'


# Save the array to a .mat file
filename_SR = 'sig_SR.mat'
scipy.io.savemat(filename_HR, {'sig_HR': sig_HR})
scipy.io.savemat(filename_rec, {'sig_rec': sig_rec})


#####################
vis.visualize_tensor(gaf_HR,'Gaf HR')
vis.visualize_tensor(gaf_SR,'Gaf SR')
vis.visualize_tensor(sampled_tensor,'Reconstructed')

vis.plot_multiple_timeseries([sig_HR, sig_SR, sig_rec], ['HR', 'SR', 'Recovered'])

#####################



