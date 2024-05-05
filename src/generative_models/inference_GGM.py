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



# # *************************
# # INFERENCE
# # *************************

# # Noise Schedule from:        NOTE: JUST USED THIS ONE FOR FASTER INFERENCE...
## ---> https://arxiv.org/pdf/2306.01875.pdf
config_diff = {
    'beta_start': 0.0001,
    'beta_end': 0.5,
    'num_steps': 10,      # Reduced number of steps
    'schedule': "quad"
}

print('Status: Inference Time...')

# ! INFERENCE NEEDS TO BE ON CPU*
inference_device = 'cpu'
device = inference_device

# Load a trained denoiser...
denoise_fun = UNet(
    in_channel=6,
    out_channel=3,
    inner_channel=64,
    norm_groups=32,
    channel_mults=(1, 2, 4, 8, 8),
    attn_res=[16],
    res_blocks=3,
    dropout=0.2,
    with_noise_level_emb=True,
    image_size=128
).to(device)      # Move the denoising model to the GPU if available

# LOAD TRAINED MODELS
save_model_dn = ''
save_model_diff = ''

# SET MODELS IN EVAL MODE
denoise_fun.load_state_dict(torch.load(save_model_dn, map_location=device))
denoise_fun.eval()
diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=3,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load(save_model_diff, map_location=device))


print('Status: Diffusion and denoising model loaded successfully')
    
# INFERENCE (NOT IN TRAINING SET)
# TODO: probably needed to put the GGM on CPU here...
print('GGM INFO')
print(type(ggm_SR))
print('Device ggm_SR', ggm_SR.device)
x = ggm_SR.to("cpu")   
x = x.to(torch.float32)
print('X info', x.device)
print(type(x))
print(x.dtype)
sampled_tensor = diffusion.p_sample_loop_single(x)
sampled_tensor = sampled_tensor.unsqueeze(0)

# Save the sampled_tensor as a pickle file... 
save_tensor_sample = 'ggm_sampled_' + str(formatted_time) + '.pkl'
with open(save_tensor_sample,'wb') as f:
    pickle.dump(sampled_tensor, f)
