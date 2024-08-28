import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Embed import DataEmbedding
from 研究生课题.时序卷积.MSK_TCN.对比实验.Layer.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self,seq_len,pred_len,d_model,num_kernels=3,top_k=5):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k      # 根据论文中的top_k个数设置
        d_ff = d_model
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model,     # 输入通道数
                               d_ff,        # 输出通道数
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff,
                               d_model,
                               num_kernels=num_kernels))

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self,seq_len,pred_len,enc_in,d_model=32):
        super(Model, self).__init__()
        self.seq_len = seq_len          # 窗口长度
        self.pred_len = pred_len        # 预测长度
        self.layer = 2   # e_layers

        self.model = nn.ModuleList([TimesBlock(seq_len,pred_len,d_model)
                                    for _ in range(self.layer)])
        self.enc_embedding = DataEmbedding(enc_in, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, 1, bias=True)      # 维度上的线性变换

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()        # 对第二个维度求均值
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)         # 对输入的特征维度进行升维[B,T,D]->[B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # 时间维度进行翻倍  align temporal dimension

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out


    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)[:, -self.pred_len:, :]       # [B, L, D]
        return dec_out.squeeze(-1)


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # [B, D, L]
    model = Model(seq_len=32,pred_len=1,enc_in=15)
    out = model(x)
    print(out.shape)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total params: {total_params}')

