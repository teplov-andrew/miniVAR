class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        encoder = []
        encoder.append(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1))
        encoder.append(nn.ReLU(inplace=True))
        encoder.append(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        encoder.append(nn.ReLU(inplace=True))
        encoder.append(nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1))
        encoder.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*encoder)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        decoder = []
        decoder.append(nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1))
        decoder.append(nn.ReLU(inplace=True))
        decoder.append(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1))
        decoder.append(nn.ReLU(inplace=True))
        decoder.append(nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1))


        self.net = nn.Sequential(*decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VQVAEModel(nn.Module):
    def __init__(
        self,
        ce_loss_scale: float = 1.0,
        latent_dim: int = 16,
        num_embeddings: int = 64,
        latent_size: tuple = (7, 7),
        levels: list[int] = [1, 2, 4, 7],
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, beta = 0.25, levels=levels)
        self.ce_loss_scale = ce_loss_scale
        self.latent_size = latent_size
        self.levels = levels


    def multi_scale_forward(self, x: torch.Tensor) -> tuple:
        B = x.size(0)
        C = self.vq_layer.embedding_dim

        z = self.encoder(x)
        q, vq_loss = self.vq_layer(z)
        idx7 = self.vq_layer.get_code_indices(z)
        
        latent_H, latent_W = self.latent_size
        canvas = torch.zeros(B, C, latent_H, latent_W, device=device)
        
        logits = {}
        for k, p in enumerate(self.levels):
            if p == self.latent_size[0]:
                idx_p = idx7
            else:
                idx_p = F.interpolate(idx7.unsqueeze(1).float(), size=(p, p), mode="nearest").long().squeeze(1)
            zq_p = self.vq_layer.get_quantized(idx_p)

            alpha = k / (len(self.levels) - 1)
            h_p = self.vq_layer.phis(zq_p, alpha)
            h_p_up = F.interpolate(h_p, size=(latent_H, latent_W), mode="bicubic")
            canvas = canvas + h_p_up

            logits[p] = self.decoder(canvas)
            
        return logits, vq_loss

    def multi_scale_loss(self, x: torch.Tensor) -> dict:

        target = x.squeeze(1).long()
        logits, vq_loss = self.multi_scale_forward(x)
        
        z = self.encoder(x)
        q, _ = self.vq_layer(z)
        logits_raw = self.decoder(q) 
        
        rec_loss = 0.0
        
        # w0 = 0.1
        # rec_loss = rec_loss + w0 * F.cross_entropy(logits_raw, target)
        # weights = {1: 0.1, 2: 0.2, 4: 0.3, 7: 0.4}

        w_raw = 0.25
        rec_loss = w_raw * F.cross_entropy(logits_raw, target)
        weights = {1: 0.05, 2: 0.15, 4: 0.30, 7: 0.50}
                 
        for p, w in weights.items():
            rec_loss = rec_loss + w * F.cross_entropy(logits[p], target)

        total = self.ce_loss_scale * rec_loss + vq_loss

        return {
            "total_loss": total,
            "recon_loss": rec_loss,
            "vq_loss": vq_loss,
        }

    def forward(self, x: torch.Tensor) -> tuple:
        
        # apply encoder -> apply vector quantizer (it returns quantized representation + vq_loss) ->
        # -> apply decoder (it returns decoded samples) 

        z = self.encoder(x)

        q, vq_loss = self.vq_layer(z)

        decoded = self.decoder(q)

        return decoded, vq_loss

    def loss(self, x: torch.Tensor) -> dict:
    
        # apply model -> get cross entropy loss

        decoded, vq_loss = self.forward(x)
        ce_loss = F.cross_entropy(decoded, x.squeeze(1).long())

        return {
            "total_loss": self.ce_loss_scale * ce_loss + vq_loss,
            "ce_loss": self.ce_loss_scale * ce_loss,
            "vq_loss": vq_loss,
        }

    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        # apply encoder -> get indices of codes using vector quantizer
        
        z = self.encoder(x)
        codebook_indices = self.vq_layer.get_code_indices(z)
        return codebook_indices

    def prior(self, n: int) -> torch.Tensor:
        # get samples from categorical distribution -> get quantized representations using vector quantizer
    
        indices = torch.randint(0, self.vq_layer.num_embeddings, (n, *self.latent_size), device="cuda:0")
        quantized = self.vq_layer.get_quantized(indices)
        return quantized

    def sample_from_logits(self, logits: torch.Tensor) -> np.ndarray:
        
        # apply softmax to the logits -> sample from the distribution
        
        probs = F.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = probs.shape
        probs_flat = probs.view(-1, C)
        samples_flat = torch.multinomial(probs_flat, num_samples=1)
        samples = samples_flat.view(B, H, W)
        samples = samples.unsqueeze(1)

        return samples.cpu().numpy()

    def sample(self, n: int) -> np.ndarray:
        with torch.no_grad():

            # sample from prior distribution -> apply decoder -> sample from logits

            quantized = self.prior(n)
            logits = self.decoder(quantized)
            samples = self.sample_from_logits(logits)
            return samples

if __name__ == "__main__":
    model = VQVAEModel().cuda()
    x = torch.zeros((2, 1, 28, 28)).cuda()

    encoded = model.encoder(x)
    size = encoded.shape[2:]
    assert size == model.latent_size

    indices = model.get_indices(x)
    assert indices.shape == (2, 7, 7)

    losses = model.loss(x)
    assert isinstance(losses, dict)
    assert "total_loss" in losses

    quantized = model.prior(10)
    assert quantized.shape == (10, 16, *model.latent_size)

    decoded = model.decoder(quantized)
    assert decoded.shape == (10, 2, 28, 28)

    sampled = model.sample(10)
    assert sampled.shape == (10, 1, 28, 28)