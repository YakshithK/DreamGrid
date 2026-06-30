import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, code_dim=64, commitment_cost=0.25):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e):
        """
        z_e: [B, D, 10, 10]
        
        returns:
            z_q_st: straight-through quantized tensor [B, D, 10, 10]
            code_ids: discrete code ids [B, 10, 10]
            vq_loss: codebook + commitment loss
            perplexity: codebook usage measure
        """
        b, d, h, w = z_e.shape

        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, d)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        code_ids = distances.argmin(dim=1)

        z_q = self.codebook(code_ids)
        z_q = z_q.view(b, h, w, d)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()

        onehot = F.one_hot(code_ids, num_classes=self.num_codes).float()
        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(
            -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
        )

        code_ids = code_ids.view(b, h, w)

        return z_q_st, code_ids, vq_loss, perplexity
    
class VQVAE(nn.Module):
    def __init__(
            self,
            num_codes=128,
            code_dim=64,
            hidden_dim=128,
            num_tile_classes=5,
            commitment_cost=0.25
    ):
        
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.num_tile_classes = num_tile_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, code_dim, kernel_size=3, padding=1)
        )

        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_cost=commitment_cost
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(code_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.rgb_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)

        self.tile_head = nn.Sequential(
            nn.Conv2d(code_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_tile_classes, kernel_size=1)
        )

    def encode(self, image):
        """
        image: [B, 3, 80, 80]
        returns code_ids: [B, 10, 10]
        """
        z_e = self.encoder(image)
        _, code_ids, _, _ = self.quantizer(z_e)
        return code_ids
    
    def encode_quantized(self, image):
        """
        image: [B, 3, 80, 80]
        returns quantized latent: [B, D, 10, 10]
        """
        z_e = self.encoder(image)
        z_q, code_ids, vq_loss, perplexity = self.quantizer(z_e)
        return z_q, code_ids, vq_loss, perplexity
    
    def decode_codes(self, code_ids):
        """
        code_ids: [B, 10, 10]
        returns:
            recon_rgb_logits: [B, 3, 80, 80]
            tile_logits: [B, 5, 10, 10]
        """
        z_q = self.quantizer.codebook(code_ids)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        h = self.decoder(z_q)
        rgb_logits = self.rgb_head(h)
        tile_logits = self.tile_head(z_q)

        return rgb_logits, tile_logits

    def forward(self, image):
        z_e = self.encoder(image)
        z_q, code_ids, vq_loss, perplexity = self.quantizer(z_e)

        h = self.decoder(z_q)
        rgb_logits = self.rgb_head(h)
        tile_logits = self.tile_head(z_q)

        return {
            "rgb_logits": rgb_logits,
            "tile_logits": tile_logits,
            "code_ids": code_ids,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "perplexity": perplexity
        }

    def decode_code_tiles(self, code_ids):
        """
        code_ids: [B, 10, 10]
        returns tile_logits: [B, 5, 10, 10]
        """
        z_q = self.quantizer.codebook(code_ids)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.tile_head(z_q)