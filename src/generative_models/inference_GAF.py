import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import device
from sklearn.model_selection import train_test_split
from datetime import datetime

from diffusion import GaussianDiffusion
from unet import UNet

from embedding import EmbeddingGAF

from visualizations import Visualizations



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

# # Noise Schedule from: https://arxiv.org/pdf/2306.01875.pdf
# config_diff = {
#     'beta_start': 0.0001,
#     'beta_end': 0.5,
#     'num_steps': 10,      # Reduced number of steps
#     'schedule': "quad"
# }

config_diff = {
    'beta_start': 1e-6,
    'beta_end': 1e-2,
    'num_steps': 2000,      # Reduced number of steps
    'schedule': "linear"
}

# saved_model_dn = 
# saved_model_diff = 


denoise_fun.load_state_dict(torch.load('saved_models/run_3/dn_model_gaf19h57.pth', map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load('saved_models/run_3/diff_model_gaf19h57.pth', map_location=device))

print('Status: Diffusion and denoising model loaded successfully')
    
############################
############################
embedding_gaf = EmbeddingGAF()

# LOAD DATA
with open('ardb_slices_clean.pkl', 'rb') as f:
    clean_signals = pickle.load(f)

sig_HR = clean_signals[52222][:128]
gaf_HR = embedding_gaf.ecg_to_GAF(sig_HR)

del clean_signals                           # REMOVE FROM MEMORY

with open('ardb_slices_noisy.pkl', 'rb') as f:
    noisy_signals = pickle.load(f)

sig_SR = noisy_signals[52222][:128]
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
# hour, minute = datetime.now().hour, datetime.now().minute
# formatted_time = f"{hour}h{minute:02d}"
# save_tensor_sample = 'gaf_sampled_' + str(formatted_time) + '.pkl'
# with open(save_tensor_sample,'wb') as f:
#     pickle.dump(sampled_tensor, f)


# RECOVER
sig_rec = embedding_gaf.GAF_to_ecg(sampled_tensor)


#####################
#####################


visualize = Visualizations()
visualize.visualize_tensor(sampled_tensor)

visualize.plot_multiple_timeseries([sig_HR, sig_SR, sig_rec], ['HR', 'SR', 'Recovered'])

#####################
#####################


