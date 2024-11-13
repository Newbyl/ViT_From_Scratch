import torch.nn as nn
import torch

class VitEmbedding(nn.Module):
    def __init__(
        self,
        image_size=224,           # Standard ImageNet size
        patch_size=16,            # Standard ViT patch size
        in_channels=3,            # RGB images
        embed_dim=768,            # ViT-Base embedding dimension
        dropout=0.1               # Standard dropout rate
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patchify and projection layer
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)

        # CLS token (learned vector that aggregates image features for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (add 1 for cls token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Dropout
        self.pos_drop = nn.Dropout(p=dropout)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.image_size, f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})"

        # Patchify and project
        # Reshape image into patches: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size * self.patch_size)

        # Project patches
        x = self.projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply dropout
        x = self.pos_drop(x)

        return x