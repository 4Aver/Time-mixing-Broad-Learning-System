import torch
import torch.nn as nn
import torch.nn.functional as F
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.SelfAttention_Family import FullAttention, AttentionLayer
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, seq_len, pred_len, enc_in, d_model=64):
        super(Model, self).__init__()
        self.pred_len = pred_len
        n_heads = 8
        e_layers = 3
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False), d_model, n_heads),
                    d_model,
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.projection = nn.Linear(d_model, pred_len, bias=True)
        self.dense_feature = nn.Linear(enc_in,1)
        # self.dense_pred = nn.Linear()

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        dec_out = self.dense_feature(dec_out).squeeze(-1)
        dec_out = dec_out[:, -self.pred_len:]  # [B, L, D]
        return dec_out


if __name__ == "__main__":
    x = torch.randn((128,15,32))        # [B, L, D]
    model = Model(seq_len=15,pred_len=1,enc_in=32)
    out = model(x)
    print(out.shape)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total params: {total_params}')


