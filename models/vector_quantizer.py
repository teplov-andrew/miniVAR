class VectorQuantizer(nn.Module):
    def __init__(
        self, num_embeddings: int = 128, embedding_dim: int = 16, beta: float = 0.25, levels = [1, 2, 4, 7]) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        
        self.mlp_for_phis = nn.ModuleList()
        for i in range(len(levels)):
            mlp = []
            mlp.append(nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1))
            mlp.append(nn.SiLU())
            mlp.append(nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1))
            self.mlp_for_phis.append(nn.Sequential(*mlp))
        

    def get_code_indices(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape[:-1]
        flattened = x.view(-1, self.embedding_dim)
        
        # calculate distances from flatten inputs to embeddings
        # find nearest embeddings to each input (use argmin op)

        distances = (torch.sum(flattened ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flattened, self.embedding.weight.t())
                     )

        # Derive the indices for minimum distances.
        encoding_indices = torch.argmin(distances, dim=1)
        
        encoding_indices = encoding_indices.view(input_shape)
        return encoding_indices

    def get_quantized(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        # get embeddgins with appropriate indices
        # transform tensor from BHWC to BCHW format
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()
        
        return quantized
        
    def phis(self, zq: torch.Tensor, stage_ratio: float) -> torch.Tensor:
        K = len(self.mlp_for_phis)
        x = stage_ratio * (K - 1)
        i_0 = int(torch.floor(torch.tensor(x)).item())
        i_1 = min(K - 1, i_0 + 1)
        delta = x - i_0

        phi_i_0 = self.mlp_for_phis[i_0](zq)
        if i_0 == i_1:
            return phi_i_0
        else:
            phi_i_1 = self.mlp_for_phis[i_1](zq)
            return (1 - delta) * phi_i_0 + delta * phi_i_1

    
    def forward(self, x: torch.Tensor) -> tuple:
        
        # get indices -> get quantized latents -> calculate codebook and commitment loss
        # final loss is codebook_loss + beta * commitment_loss

        quantized = self.get_quantized(self.get_code_indices(x))

        loss = torch.mean((quantized.detach() - x)**2) + self.beta * torch.mean((quantized - x.detach())**2)

        # Straight-through estimator!!! 
        quantized = x + (quantized - x).detach()

        return quantized, loss


if __name__ == "__main__":
    x = torch.zeros((1, 16, 7, 7))
    layer = VectorQuantizer()
    indices = layer.get_code_indices(x)
    assert indices.shape == (1, 7, 7)
    quantized = layer.get_quantized(indices)
    assert quantized.shape == (1, 16, 7, 7)
    quantized, loss = layer(x)
    assert quantized.shape == (1, 16, 7, 7)
    assert loss.shape == ()