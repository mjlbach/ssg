import math

import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        # 3 layer embedding network for obj feature vectors
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
        )

        # The CLS Token used as input on the first element of each sequence
        self.cls = nn.parameter.Parameter(
            data=torch.nn.init.xavier_uniform_(torch.zeros(1, num_features), gain=1.0)
        )

        # Transformer encoder layer which oeprates on node embeddings
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding dimension of model
        self.d_model = d_model

        # Out features after decoding
        self.out_features = ntoken

        # Decoder which reads out CLS token
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # Make mask based on repeated sequence length, padding with 1 for CLS token (appended at start of sequence)
        # Mask == True means that the elements per each sequence will *not* be included in the attention mask
        mask = torch.arange(src.shape[1] + 1).repeat((src.shape[0], 1)).to(
            lengths.device
        ) >= (lengths[:, None] + 1)

        # Append CLS token to start of sequence
        cls = torch.broadcast_to(
            self.cls, (src.shape[0], self.cls.shape[0], self.cls.shape[1])
        )
        src = torch.cat([cls, src], dim=1)

        # Run the model
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.decoder(output)

        # Read out only CLS token
        output = output[:, 0, :]

        return output


def main():
    data = torch.randn((5, 4, 8))
    idx = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
    model = TransformerModel(
        num_features=8,
        ntoken=128,
        d_model=128,
        d_hid=200,
        nhead=8,
        dropout=0.2,
        nlayers=2,
    )
    out = model(data, idx)


if __name__ == "__main__":
    main()
