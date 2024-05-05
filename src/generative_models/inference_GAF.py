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
from embedding import EmbeddingGGM




print('Status: Inference Time...')

# ! INFERENCE NEEDS TO BE ON CPU
inference_device = 'cpu'
device = inference_device

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






denoise_fun.load_state_dict(torch.load(save_model_dn, map_location=device))
denoise_fun.eval()

diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load(save_model_diff, map_location=device))

print('Status: Diffusion and denoising model loaded successfully')
    
# INFERENCE (NOT IN TRAINING SET)
# TODO: convert gaf_SR to  Input type (double) and bias type (float) should be the same ... float32
gaf_SR = gaf_SR.float()
sampled_tensor = diffusion.p_sample_loop_single(gaf_SR)
sampled_tensor = sampled_tensor.unsqueeze(0)

# Save the sampled_tensor as a pickle file... 
save_tensor_sample = 'gaf_sampled_' + str(formatted_time) + '.pkl'
with open(save_tensor_sample,'wb') as f:
    pickle.dump(sampled_tensor, f)

