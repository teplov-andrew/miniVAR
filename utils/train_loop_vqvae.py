from collections import defaultdict
from tqdm.auto import tqdm
from typing import Tuple

import torch
from torch import optim
import wandb


def train_epoch(
    model: object,
    train_loader: object,
    optimizer: object,
    use_cuda: bool,
    loss_key: str = "total",
) -> defaultdict:
    model.train()

    stats = defaultdict(list)
    for x in train_loader:
        if use_cuda:
            x = x.cuda()    
        losses = model.multi_scale_loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model: object, data_loader: object, use_cuda: bool) -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in data_loader:
            if use_cuda:
                x = x.cuda()
            losses = model.multi_scale_loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats


def train_model(
    model: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    lr: float,
    use_tqdm: bool = False,
    use_cuda: bool = False,
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()

    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model, test_loader, use_cuda)

        train_loss_epoch = train_loss["total_loss"][-1]    
        test_loss_epoch  = test_loss["total_loss"]       
        
        wandb.log({
            "loss/train_total": train_loss_epoch,
            "loss/train_recon": train_loss["recon_loss"][-1],
            "loss/train_vq":    train_loss["vq_loss"][-1],
            "loss/test_total":  test_loss["total_loss"],
            "loss/test_recon":  test_loss["recon_loss"],
            "loss/test_vq":     test_loss["vq_loss"],
        }, step=epoch)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
            
        # wandb.log({"loss_train_total_loss": dict(train_losses)["total_loss"], "epoch": epoch})
        # wandb.log({"loss_train_recon_loss": dict(train_losses)["recon_loss"], "epoch": epoch})
        # wandb.log({"loss_train_vq_loss": dict(train_losses)["vq_loss"], "epoch": epoch})

        # wandb.log({"loss_test_total_loss": dict(test_losses)["total_loss"], "epoch": epoch})
        # wandb.log({"loss_test_recon_loss": dict(test_losses)["recon_loss"], "epoch": epoch})
        # wandb.log({"loss_test_vq_loss": dict(test_losses)["vq_loss"], "epoch": epoch})
        
    return dict(train_losses), dict(test_losses)