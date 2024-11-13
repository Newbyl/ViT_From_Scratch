import torch
import torch.nn as nn
from encoder import Encoder
from embeddings import VitEmbedding

class ViT(nn.Module):
    def __init__(self, embed_size=768, num_layers=12, heads=12, device='cuda', num_class=10):
        super(ViT, self).__init__()
        self.src_pad_idx = 0
        self.device = device

        self.encoder = Encoder(heads=heads,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               device=device)

        self.patch_embed = VitEmbedding(embed_dim=embed_size)

        self.clas_head = nn.Linear(embed_size, num_class)

        self.softmax = nn.Softmax(1)


    def forward(self, images):
        assert len(images.shape) == 4, 'The input is not a batch'
        embedded = self.patch_embed(images)
        hidden_state = self.encoder(embedded)
        logits = self.clas_head(hidden_state[:, 0, :])

        return logits

    def predict_proba(self, images):
        logits = self.forward(images)

        return torch.softmax(logits, dim=-1)