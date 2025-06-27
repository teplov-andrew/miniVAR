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

from models.vq_vae import VQVAEModel
from utils.train_loop_vqvae import train_model
from utils.load_dataset import load_dataset
from utils.visualize import vq_vae_visulization, show_samples

from config import CFG

USE_CUDA = torch.cuda.is_available()

print("cuda is available:", USE_CUDA)
if USE_CUDA:
    device = "cuda"
else:
    device = "cpu"

BATCH_SIZE = CFG.BATCH_SIZE
EPOCHS = CFG.VQVAE_EPOCHS
LR = CFG.VQVAE_LR
CE_SCALE = CFG.VQVAE_CE_SCALE

wandb.init(
    project=CFG.WANDB_PROJECT,  
    name="VQ-VAE_training",
    entity=CFG.WANDB_ENTITY,   
    config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
)

train_data, test_data = load_dataset("mnist", flatten=False, binarize=True)

model = VQVAEModel(ce_loss_scale=CE_SCALE, latent_dim=CFG.VQVAE_LATENT_DIM, num_embeddings=CFG.VQVAE_NUM_EMBEDDINGS)

train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE)

train_losses, test_losses = train_model(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    use_cuda=USE_CUDA,
    use_tqdm=True,
    lr=LR,
)


if not os.path.exists(CFG.CHECKPOINTS_PATH):
    os.makedirs(CFG.CHECKPOINTS_PATH)
    
save_path = CFG.CHECKPOINTS_PATH + "/" + "vqvae_model_weights.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

vq_vae_visulization(model, test_loader, test_losses, show_samples)
