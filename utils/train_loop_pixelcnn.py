import torch
import torch.nn.functional as F
from torch import optim
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
from config import CFG
from typing import Tuple
from models.pixel_cnn import PixelCNN_NX

def train_epoch(
    model: object,
    train_loader: object,
    optimizer: object,
    use_cuda: bool,
    loss_key: str = "total",
) -> defaultdict:
    model.train()

    stats = defaultdict(list)
    for batch in train_loader:
        if use_cuda:
            batch = {k: v.cuda() for k, v in batch.items()}

        for i in range(len(CFG.LEVELS)-1):
            prev = batch[CFG.LEVELS[i]]
            curr = batch[CFG.LEVELS[i+1]]
            stage_id = CFG.STAGE_IDS[CFG.LEVELS[i+1]]
            
            logits = model(prev, stage_id, out_shape=(CFG.LEVELS[i+1], CFG.LEVELS[i+1]))
            loss = F.cross_entropy(logits.squeeze(2).squeeze(1), curr.squeeze(1).long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats[f"loss_{CFG.LEVELS[i+1]}"] .append(loss.item())

    return stats


def eval_model(model: object, data_loader: object, use_cuda: bool) -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            if use_cuda:
                batch = {k: v.cuda() for k, v in batch.items()}
            bs = next(iter(batch.values())).size(0)
            total += bs
            for i in range(len(CFG.LEVELS)-1):
                prev = batch[CFG.LEVELS[i]]
                curr = batch[CFG.LEVELS[i+1]]
                stage_id = CFG.STAGE_IDS[CFG.LEVELS[i+1]]
                logits = model(prev, stage_id, out_shape=(CFG.LEVELS[i+1], CFG.LEVELS[i+1]))
                loss = F.cross_entropy(logits.squeeze(2).squeeze(1), curr.squeeze(1).long())
                stats[f"loss_{CFG.LEVELS[i+1]}" ] += loss.item() * bs
                
        for k in stats:
            stats[k] /= total
    return stats


def sample_nsp(model: PixelCNN_NX,
               levels: list[int] = [1, 2, 4, 7],
               stage_ids: dict = {1: 0, 2: 1, 4: 2, 7: 3},
               num_samples: int = 8,
               cfg_scale: float = 6.5,
               device="cuda",) -> torch.LongTensor:
    samples = {levels[0]: torch.randint(
        model.num_embeddings, 
        size=(num_samples, 1, levels[0], levels[0]),
        device=device,
        dtype=torch.long
)}
    with torch.no_grad():

        for i in range(1, len(levels)):
            prev_level = levels[i - 1]
            curr_level = levels[i]
            stage_id = stage_ids[curr_level]
            out_shape = (curr_level, curr_level)
    
            curr_sample = torch.zeros(num_samples, 1, *out_shape, dtype=torch.long, device=device)
    
            for r in range(curr_level):
                for c in range(curr_level):
                    # logits=model(samples[prev_level], stage_id, out_shape).squeeze(2)
                    inp = samples[prev_level].repeat(2,1,1,1)
                    logits2 = model(inp, stage_id, out_shape).squeeze(2)
                    logits_c, logits_u = logits2.chunk(2, 0)
                    logits = logits_u + cfg_scale*(logits_c - logits_u)
                    
                    probs = F.softmax(logits[:, :, r, c], dim=1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    curr_sample[:, 0, r, c] = token
            samples[curr_level] = curr_sample
        
    return samples, samples[levels[-1]]

@torch.no_grad()
def visualize_nsp(model_nsp, vq_vae, levels, stage_ids, num_samples=8, device="cuda"):
    model_nsp.eval()
    
    samples_dict = {levels[0]: torch.randint(
        model_nsp.num_embeddings, 
        size=(num_samples, 1, levels[0], levels[0]),
        device=device,
        dtype=torch.long
    )}
    
    for i in range(1, len(levels)):
        prev_level = levels[i - 1]
        curr_level = levels[i]
        stage_id = stage_ids[curr_level]
        out_shape = (curr_level, curr_level)

        curr_sample = torch.zeros(num_samples, 1, *out_shape, dtype=torch.long, device=device)

        for r in range(curr_level):
            for c in range(curr_level):
                logits = model_nsp(samples_dict[prev_level], stage_id, out_shape).squeeze(2)
                probs = F.softmax(logits[:, :, r, c], dim=1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                curr_sample[:, 0, r, c] = token
        samples_dict[curr_level] = curr_sample

    rep_imgs = []
    latent_H, latent_W = vq_vae.latent_size
    for p in levels:
        idx = samples_dict[p].squeeze(1)
        quantized = vq_vae.vq_layer.get_quantized(torch.Tensor(idx))

        quantized = F.interpolate(
            quantized,
            size=(latent_H,latent_W),
            mode="bicubic"
        )
        logits = vq_vae.decoder(quantized)
        samples = vq_vae.sample_from_logits(logits)
        
        samples = samples.astype("float32")
        rep_imgs.append(samples[0].squeeze(0))
        # show_samples(samples, title=f"NSP @ {p}×{p}")
        
    fig, axes = plt.subplots(1, len(levels), 
                             figsize=(len(levels) * 3, 3))
    for ax, img, p in zip(axes, rep_imgs, levels):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"NSP @ {p}×{p}")
    plt.tight_layout()
    plt.show()


def train_model(
    model_nsp: object,
    vq_vae: object,
    levels, 
    stage_ids,
    train_loader: object,
    test_loader: object,
    epochs: int,
    lr: float,
    use_tqdm: bool = False,
    use_cuda: bool = False,
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:
    optimizer = optim.Adam(model_nsp.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model_nsp = model_nsp.cuda()

    for epoch in forrange:
        model_nsp.train()
        train_loss = train_epoch(model_nsp, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model_nsp, test_loader, use_cuda)

        # print(train_loss)
        train_loss_epoch = train_loss["loss_7"][-1]    
        test_loss_epoch  = test_loss["loss_7"]        
        
        wandb.log({
            "loss/train_7": train_loss_epoch,
            "loss/train_4": train_loss["loss_4"][-1],
            "loss/train_2":    train_loss["loss_2"][-1],
            "loss/test_7":  test_loss["loss_7"],
            "loss/test_4":  test_loss["loss_4"],
            "loss/test_2":     test_loss["loss_2"],
        }, step=epoch)

        if (epoch + 1) % 5 == 0:
            print("EPOCH:", epoch + 1)
            save_path_nsp = f"pixelcnn_nsp_epoch_{epoch+1}.png"
            fig = plt.figure(figsize=(len(levels) * 3, 3))
            visualize_nsp(model_nsp, vq_vae, levels, stage_ids)
            plt.savefig(CFG.VISUALIZETIONS_PATH + "/" + save_path_nsp)
            plt.close(fig)
            print(f"NSP visualization saved to {save_path_nsp}")

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return dict(train_losses), dict(test_losses)