import torch
import scipy.io
from torch import device

from diffusion import GaussianDiffusion
from unet import UNet
from embedding import EmbeddingGAF

from visualizations import Visualizations
from datahelper import DataHelper


#################################
# AUX METHODS
vis = Visualizations()
embedding_gaf = EmbeddingGAF()
dl = DataHelper()

#################################
# CONFIGURE AND LOAD MODEL
device = 'cpu'

# Parameters of the U-Net (denoising function)
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
).to(device)  

# Noise schedule (from GitHub)
config_diff = {
    'beta_start': 1e-6,
    'beta_end': 1e-2,
    'num_steps': 2000,      
    'schedule': "linear"
}

# Load Pre-Trained Model
denoise_fun.load_state_dict(torch.load('models/dn_model_1.pth', map_location=device))
denoise_fun.eval()
diffusion = GaussianDiffusion(denoise_fun, image_size=(128,128),channels=1,loss_type='l1',conditional=True,config_diff=config_diff).to(device)  # Move the diffusion model to the GPU if available
diffusion.load_state_dict(torch.load('models/diff_model_1.pth', map_location=device))

print('Status: Diffusion and denoising model loaded successfully')

#################################
# LOAD SAMPLES
signals_HR , signals_SR = dl.load_data_from_directory('samples/noisy_samples', 'samples/clean_samples/af_sig_HR.mat', 'samples/clean_samples/ardb_sig_HR.mat')

num_of_shots = 1

for i in range(len(signals_HR)):

    gaf_HR = embedding_gaf.ecg_to_GAF(signals_HR[i])
    gaf_SR = embedding_gaf.ecg_to_GAF(signals_SR[i])

    ############################
    # INFERENCE 
    for j in range(num_of_shots):

        print('Sampling... run', i)

        # FLOAT.32
        x = gaf_SR.to("cpu")   
        x = x.to(torch.float32)

        # SAMPLE TENSOR
        sampled_tensor = diffusion.p_sample_loop_single(x)
        sampled_tensor = sampled_tensor.unsqueeze(0)

        # RECOVER SIGNAL
        sig_rec = embedding_gaf.GAF_to_ecg(sampled_tensor)
        
        # SAVE
        filename_rec =  str(i) + 'sig_rec_' + str(j) + '.mat'

        print('Saved as:', filename_rec)

        # Save the array to a .mat file
        scipy.io.savemat(filename_rec, {'sig_rec': sig_rec})



