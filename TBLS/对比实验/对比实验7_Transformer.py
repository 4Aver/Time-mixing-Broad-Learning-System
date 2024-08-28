import torch
import torch.nn as nn
import torch.nn.functional as F
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.SelfAttention_Family import FullAttention, AttentionLayer
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self,seq_len,pred_len,enc_in,d_model=32):
        super(Model, self).__init__()
        self.pred_len = pred_len
        e_layers = 2
        n_heads = 8
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False), d_model, n_heads),
                    d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.dense_feature = nn.Linear(d_model,1)

    def forecast(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.dense_feature(enc_out)
        return enc_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]


if __name__ == "__main__":
    x = torch.randn((128,15,32))        # [B, L, D]
    model = Model(seq_len=15,pred_len=1,enc_in=32)
    out = model(x)
    print(out.shape)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total params: {total_params}')
