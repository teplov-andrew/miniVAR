import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from typing import Dict, List, Optional, Tuple
from models.vq_vae import VQVAEModel

TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16


def plot_training_curves(
    train_losses: Dict[str, np.ndarray],
    test_losses: Dict[str, np.ndarray],
    logscale_y: bool = False,
    logscale_x: bool = False,
) -> None:
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)

    plt.figure()
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + "_train")

    for key, value in test_losses.items():
        plt.plot(x_test, value, label=key + "_test")

    if logscale_y:
        plt.semilogy()

    if logscale_x:
        plt.semilogx()

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.grid()
    plt.show()


def show_samples(
    samples: np.ndarray,
    title: str,
    figsize: Optional[Tuple[int, int]] = None,
    nrow: Optional[int] = None,
) -> None:
    if isinstance(samples, np.ndarray):
        samples = torch.FloatTensor(samples)
    if nrow is None:
        nrow = int(np.sqrt(len(samples)))
    grid_samples = make_grid(samples, nrow=nrow)

    grid_img = grid_samples.permute(1, 2, 0)
    if figsize is None:
        figsize = (6, 6)

    grid_img = grid_img.clip(0, 1)
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()


def visualize_images(data: np.ndarray, title: str) -> None:
    idxs = np.random.choice(len(data), replace=False, size=(100,))
    images = data[idxs]
    show_samples(images, title)


def visualize_2d_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    train_labels: Optional[str] = None,
    test_labels: Optional[str] = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("train", fontsize=TITLE_FONT_SIZE)
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.tick_params(labelsize=LABEL_FONT_SIZE)
    ax2.set_title("test", fontsize=TITLE_FONT_SIZE)
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax2.tick_params(labelsize=LABEL_FONT_SIZE)
    plt.show()


def visualize_2d_samples(
    data: np.ndarray,
    title: str,
    labels: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], s=1, c=labels)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()


def visualize_2d_densities(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    densities: np.ndarray,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    densities = densities.reshape([y_grid.shape[0], y_grid.shape[1]])
    plt.figure(figsize=(5, 5))
    plt.pcolor(x_grid, y_grid, densities)
    plt.pcolor(x_grid, y_grid, densities)

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()




def vq_vae_visulization(model, test_loader, test_losses, show_samples, save_path_samples="vqvae_samples.png", save_path_recon="vqvae_reconstructions.png"):
    for key, value in test_losses.items():
        print("{}: {:.4f}".format(key, value[-1]))

    samples = model.sample(100)
    samples = samples.astype("float32")
    # show_samples(samples, title="Samples")
    plt.savefig(save_path_samples)
    print(f"Samples image saved to {save_path_samples}")

    x = next(iter(test_loader))[:50].cuda()
    with torch.no_grad():
        decoded, _ = model(x)
        x_recon = model.sample_from_logits(decoded)
    x = x.cpu().numpy()
    reconstructions = np.concatenate((x, x_recon), axis=0)
    reconstructions = reconstructions.astype("float32")
    # show_samples(reconstructions, title="Reconstructions")
    plt.savefig(save_path_recon)
    print(f"Reconstructions image saved to {save_path_recon}")
    
    

@torch.no_grad()
def visualize_coarse_to_fine_samples(
    model_nsp,
    vq_vae,
    levels: list[int],
    stage_ids: dict[int,int],
    num_samples: int = 4,
    device: str = "cuda"
):

    model_nsp.eval()
    device = torch.device(device)
    
    samples_idx = {
        levels[0]: torch.randint(
            0,
            model_nsp.num_embeddings,
            (num_samples, 1, levels[0], levels[0]),
            device=device,
            dtype=torch.long
        )
    }
    for i in range(1, len(levels)):
        prev_p = levels[i-1]
        p      = levels[i]
        stage  = stage_ids[p]
        out_hw = (p, p)

    
        logits = model_nsp(
            samples_idx[prev_p],
            stage_id=stage,
            out_shape=out_hw
        ).squeeze(2)  # (B, V, p, p)

        curr = torch.zeros((num_samples,1,p,p), device=device, dtype=torch.long)
        for r in range(p):
            for c in range(p):
                probs    = F.softmax(logits[:,:,r,c], dim=1)      # (B, V)
                curr[:,0,r,c] = torch.multinomial(probs, 1).squeeze(-1)
        samples_idx[p] = curr


    fig, axes = plt.subplots(
        num_samples,
        len(levels)*2,
        figsize=(2*len(levels), 2*num_samples),
        squeeze=False
    )

    latent_H, latent_W = vq_vae.latent_size

    for i in range(num_samples):
        for l_idx, p in enumerate(levels):
            
            ax = axes[i][2*l_idx]
            grid = samples_idx[p][i,0].cpu().numpy() 
            ax.imshow(grid, cmap="tab20")
            ax.set_title(f"{p}×{p}")
            ax.axis("off")


            idx = samples_idx[p][i:i+1,0]            
            zq = vq_vae.vq_layer.get_quantized(torch.Tensor(idx).int().cuda())              

        
            zq_up = F.interpolate(zq, size=(latent_H,latent_W), mode="bicubic")
            recon_logits = vq_vae.decoder(zq_up)      
            img = vq_vae.sample_from_logits(recon_logits)[0,0] 

            ax = axes[i][2*l_idx+1]
            ax.imshow(img, cmap="gray")
            ax.set_title("decoded")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
from utils.train_loop_pixelcnn import sample_nsp


def show_pixelcnn_samples(model, prior_model, num_samples=100, device="cuda"):
    sampled_indices = sample_nsp(prior_model, num_samples=num_samples, device=device)[1].squeeze(1)
    quantized = model.vq_layer.get_quantized(torch.Tensor(sampled_indices).int().to(device))
    logits = model.decoder(quantized)
    samples = model.sample_from_logits(logits)
    samples = samples.astype("float32")
    show_samples(samples, title="Samples")
    

def generate_var_style_samples(model, prior_model, levels, num_samples=100, device="cuda"):
    samples_dict = sample_nsp(prior_model, num_samples=num_samples, device=device)[0]

    B = samples_dict[1].shape[0]
    C = model.vq_layer.embedding_dim
    latent_H, latent_W = model.latent_size 

    canvas = torch.zeros(B, C, latent_H, latent_W, device=device)

    for k, p in enumerate(levels):
        idx = samples_dict[p][:,0]

        zq = model.vq_layer.get_quantized(idx)

        ratio = k/(len(levels)-1)
        h_p   = model.vq_layer.phis(zq, ratio)

        h_p_up = F.interpolate(h_p, size=(latent_H, latent_W), mode="bicubic")
        canvas  = canvas + h_p_up       

    recon_logits = model.decoder(canvas)  
    samples = model.sample_from_logits(recon_logits)

    samples = samples.astype("float32")
    show_samples(samples, title="VAR-style Samples")
    



def add_level_to_canvas(idx_b1hw: torch.LongTensor,
                        stage_ratio: float,
                        vqvae: VQVAEModel,
                        canvas_bChw: torch.Tensor,
                        target_hw: tuple[int, int]) -> torch.Tensor:

    zq = vqvae.vq_layer.get_quantized(idx_b1hw.squeeze(1))      # (B,C,h,w)
    h  = vqvae.vq_layer.phis(zq, stage_ratio)                   # (B,C,h,w)
    H, W = target_hw
    h_up = F.interpolate(h, size=(H, W), mode="bicubic")
    return canvas_bChw + h_up



@torch.no_grad()
def sample_var_style(model_nsp, vqvae,
                     levels=[1,2,4,7],
                     stage_ids={1:0,2:1,4:2,7:3},
                     cfg_scale=3.0,
                     num_samples=16,
                     device="cuda"):
    
    device = torch.device(device)
    B = num_samples
    C = vqvae.vq_layer.embedding_dim
    H, W = vqvae.latent_size

    idx = {levels[0]: torch.randint(model_nsp.num_embeddings,
                                    (B,1,1,1), device=device)}

    canvas = torch.zeros(B, C, H, W, device=device)

    for i in range(1, len(levels)):
        p        = levels[i]
        prev_p   = levels[i-1]
        stage_id = stage_ids[p]

        logits_cond   = model_nsp(idx[prev_p],        stage_id, (p,p)).squeeze(2)
        logits_uncond = model_nsp(idx[prev_p]*0,      stage_id, (p,p)).squeeze(2)

        logits = logits_uncond + cfg_scale*(logits_cond - logits_uncond)

        idx_p = torch.zeros(B,1,p,p, device=device, dtype=torch.long)
        for r in range(p):
            for c in range(p):
                probs            = F.softmax(logits[:,:,r,c], dim=1)
                idx_p[:,0,r,c]   = torch.multinomial(probs, 1).squeeze(-1)
        idx[p] = idx_p

        ratio   = i/(len(levels)-1)
        canvas  = add_level_to_canvas(idx_p, ratio, vqvae, canvas, (H,W))

    recon_logits = vqvae.decoder(canvas)
    imgs  = vqvae.sample_from_logits(recon_logits)       # numpy (B,1,28,28)
    return imgs.astype("float32")
