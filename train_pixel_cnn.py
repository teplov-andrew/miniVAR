import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict
from typing import Tuple

import wandb

from models.pixel_cnn import PixelCNN_NX
from utils.train_loop_pixelcnn import train_model
from config import CFG
from utils.visualize import visualize_coarse_to_fine_samples, show_pixelcnn_samples, generate_var_style_samples, sample_var_style, show_samples

USE_CUDA = torch.cuda.is_available()

print("cuda is available:", USE_CUDA)
if USE_CUDA:
    device = "cuda"
else:
    device = "cpu"
    

from models.vq_vae import VQVAEModel


CE_SCALE = CFG.VQVAE_CE_SCALE
vqvae_model = VQVAEModel(ce_loss_scale=CE_SCALE, latent_dim=CFG.VQVAE_LATENT_DIM, num_embeddings=CFG.VQVAE_NUM_EMBEDDINGS)
vqvae_weights_path = "vqvae_model_weights.pth"
vqvae_model.load_state_dict(torch.load(vqvae_weights_path, map_location=device))
vqvae_model = vqvae_model.to(device)
vqvae_model.eval()
print(f"Loaded VQ-VAE weights from {vqvae_weights_path}")


INPUT_SHAPE = CFG.PIXELCNN_INPUT_SHAPE


train_indices_list = []
test_indices_list = []

with torch.no_grad():
    for x in train_loader:
        x = x.cuda()
        indices = vqvae_model.get_indices(x)
        indices = indices.unsqueeze(1)
        train_indices_list.append(indices.cpu().numpy())

train_indices = np.concatenate(train_indices_list, axis=0)

with torch.no_grad():
    for x in test_loader:
        x = x.cuda()
        indices = vqvae_model.get_indices(x)
        indices = indices.unsqueeze(1)
        test_indices_list.append(indices.cpu().numpy())

test_indices = np.concatenate(test_indices_list, axis=0)


assert isinstance(train_indices, np.ndarray)
assert isinstance(test_indices, np.ndarray)
assert train_indices.shape == (60000, 1, *INPUT_SHAPE)
assert test_indices.shape == (10000, 1, *INPUT_SHAPE)


def downsample(idx_7x7: torch.LongTensor, size: int) -> torch.LongTensor:
    x = idx_7x7.unsqueeze(0).float()     # (1,1,7,7)
    down = F.interpolate(x, size=(size, size), mode='nearest')
    return down.long().squeeze(0)  
    
class MultiScaleIndices(Dataset):
    def __init__(self, indices_7x7_np: np.ndarray):
        self.idx7 = torch.from_numpy(indices_7x7_np).long()

    def __len__(self):
        return len(self.idx7)

    def __getitem__(self, i):
        idx7 = self.idx7[i]               # 1×7×7
        # [1x1 -> 2x2 -> 4x4 -> 7x7]
        idx4 = downsample(idx7, size=4)   # 1×4×4
        idx2 = downsample(idx7, size=2)   # 1×2×2
        idx1 = downsample(idx7, size=1)   # 1×1×1
        return {
            1: idx1,  # 1×1×1
            2: idx2,  # 1×2×2
            4: idx4,  # 1×4×4
            7: idx7,  # 1×7×7
        }

EPOCHS = CFG.PIXELCNN_EPOCHS
BATCH_SIZE = CFG.BATCH_SIZE
LR = CFG.PIXELCNN_LR
N_LAYERS = CFG.PIXELCNN_N_LAYERS
N_FILTERS = CFG.PIXELCNN_N_FILTERS


prior_model = PixelCNN_NX(num_embeddings=128, 
                          input_shape=INPUT_SHAPE, 
                          n_filters=N_FILTERS, 
                          kernel_size=5, 
                          n_layers=N_LAYERS
)

wandb.init(
    project="VAR",  
    name="PixelCNN_training",
    entity="andrew_tep",   
    config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
)


train_loader = data.DataLoader(MultiScaleIndices(train_indices), batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(MultiScaleIndices(test_indices), batch_size=BATCH_SIZE)
train_losses, test_losses = train_model(
    prior_model,
    vqvae_model,
    CFG.LEVELS,
    CFG.STAGE_IDS,
    train_loader,
    test_loader,
    epochs=CFG.PIXELCNN_EPOCHS,
    lr=CFG.PIXELCNN_LR,
    use_tqdm=True,
    use_cuda=USE_CUDA,
)

model_save_path = "pixelcnn_model.pth"
torch.save(prior_model.state_dict(), model_save_path)
print(f"PixelCNN model saved to {model_save_path}")


visualize_coarse_to_fine_samples(
    model_nsp=prior_model,
    vq_vae=vqvae_model,
    levels=CFG.LEVELS,
    stage_ids=CFG.STAGE_IDS,
    num_samples=6,    
    device="cuda"
)
plt.savefig("pixelcnn_coarse_to_fine_samples.png")
plt.close()

plt.figure()
show_pixelcnn_samples(vqvae_model, prior_model, num_samples=36, device="cuda")
plt.savefig("pixelcnn_samples.png")
plt.close()


plt.figure()
generate_var_style_samples(vqvae_model, prior_model, CFG.LEVELS, num_samples=36, device="cuda")
plt.savefig("pixelcnn_var_style_samples.png")
plt.close()

imgs = sample_var_style(prior_model, vqvae_model, levels=CFG.LEVELS, stage_ids=CFG.STAGE_IDS, cfg_scale=CFG.PIXELCNN_CFG_SCALE, num_samples=16, device="cuda")
show_samples(imgs, title="VAR-style Sampled Images")
plt.savefig("pixelcnn_var_style_samples_cfg.png")
plt.close()


