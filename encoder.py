from transformer import TransformerBlock
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        embed_size=768,
        num_layers=12,
        heads=8,
        device='cuda',
        dropout=0.1,
        forward_expansion=4,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        #self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        #self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )"""

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(x, x, x, mask)

        return out