import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class MaskedConv2d(nn.Conv2d):
    def __init__(
        self, mask_type: str, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        assert mask_type in ["A", "B"]
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(input, self.weight * self.mask, self.bias, padding=self.padding)

    def create_mask(self, mask_type: str) -> None:
        k = self.kernel_size[0]
        self.mask[:, :, : k // 2] = 1
        self.mask[:, :, k // 2, : k // 2] = 1
        if mask_type == "B":
            self.mask[:, :, k // 2, k // 2] = 1


def test_masked_conv2d():
    layer = MaskedConv2d("A", 2, 2)
    assert np.allclose(layer.mask[:, :, 2, 2].numpy(), np.zeros((2, 2)))

    layer = MaskedConv2d("B", 2, 2)
    assert np.allclose(layer.mask[:, :, 2, 2].numpy(), np.ones((2, 2)))


test_masked_conv2d()


class PixelCNN_NX(nn.Module):
    def __init__(
        self, # нет vocab
        num_embeddings: int = 128,
        input_shape: tuple = (7,7), # current lavel size [1x1 -> 2x2 -> 4x4 -> 7x7]
        n_filters: int = 32,
        kernel_size: int = 5,
        n_layers: int = 5,
        num_stages: int = 4,
    ) -> None:

        super().__init__()
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        
        # self.stage_embed = nn.Embedding(num_stages, n_filters)
        # self.in_proj = nn.Conv2d(num_embeddings, n_filters, 1)
        
        C_embed = 64                   
        self.token_embed = nn.Embedding(num_embeddings, C_embed)
        self.stage_embed = nn.Embedding(num_stages, n_filters)
        self.in_proj = nn.Conv2d(C_embed, n_filters, 1)

        layers = []
        layers.append(MaskedConv2d("A", n_filters, n_filters, kernel_size=kernel_size)) # num_embeddings!=n_filters

        for i in range(n_layers):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2d("B", n_filters, n_filters, kernel_size=kernel_size))

        layers.extend(
            [
                nn.ReLU(),
                MaskedConv2d("B", in_channels=n_filters, out_channels=num_embeddings, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)


    def forward(self, prev_x_b1hw, stage_id: int, out_shape: Tuple[int, int]):
        h,w = out_shape
        x_upscale_b1hw = F.interpolate(prev_x_b1hw.float(), (h,w), mode="nearest").long()

        # flattened = x_upscale_b1hw.view((-1, 1))
        # encodings = torch.zeros(flattened.shape[0], self.num_embeddings).cuda()
        # encodings.scatter_(1, flattened, 1)
        # encodings = encodings.view((-1, h, w, self.num_embeddings))
        # encodings = encodings.permute((0, 3, 1, 2))

        encodings = self.token_embed(x_upscale_b1hw.squeeze(1))        # B×H×W×C_e
        encodings = encodings.permute(0,3,1,2).contiguous()  

        #stage embedding
        b = encodings.size(0)
        # emb = self.stage_embed(torch.tensor([stage_id], device=encodings.device)).view(1, -1, 1, 1)
        stage_id_tensor = torch.full((b,), stage_id, device=encodings.device, dtype=torch.long)
        emb = self.stage_embed(stage_id_tensor).view(b, -1, 1, 1)
        encodings = self.in_proj(encodings) + emb
        
        out = self.net(encodings)
        out = out.view(-1, self.num_embeddings, 1, h, w)
        return out