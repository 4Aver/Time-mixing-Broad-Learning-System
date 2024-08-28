import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers

# def batch_norm(x):
#     x = tf.convert_to_tensor(x.detach().numpy())
#     x = layers.BatchNormalization(axis=[-2, -1])(x)  # [Batch, Input Length, Channel]
#     x = torch.from_numpy(x.numpy())
#     return x


class Mixing(nn.Module):
    def __init__(self,input_dim,time_len):
        super(Mixing, self).__init__()
        FF_Dim = 3
        self.identity = nn.Identity()

        # Temporal Linear
        self.fnn_t = nn.Linear(time_len,1)
        self.relu_t = nn.ReLU()
        self.drop_t = nn.Dropout(p=0.05)

        # Feature Linear
        self.fnn1_f = nn.Linear(input_dim,FF_Dim)
        self.relu_f = nn.ReLU()
        self.drop1_f = nn.Dropout(p=0.05)
        self.fnn2_f = nn.Linear(FF_Dim,input_dim)
        self.drop2_f = nn.Dropout(p=0.05)
        self.bn = nn.BatchNorm2d(time_len,input_dim)

    def forward(self,x):
        input_x = self.identity(x)      # [Batch, Input Length, Channel]
        x = self.bn(x.unsqueeze(-1)).squeeze(-1)
        x = x.permute(0,2,1)       # [Batch, Channel, Input Length]
        x = self.fnn_t(x)
        x = self.relu_t(x)
        x = self.drop_t(x)
        x = x.permute(0,2,1)       # [Batch, Input Length, Channel]
        res = x + input_x

        x = self.bn(res.unsqueeze(-1)).squeeze(-1)
        x = self.fnn1_f(x)
        x = self.relu_f(x)
        x = self.drop1_f(x)
        x = self.fnn2_f(x)
        x = self.drop2_f(x)
        x = x + res
        return x


class TSMixer(nn.Module):
    def __init__(self,input_dim,time_len,n_block):
        super(TSMixer, self).__init__()
        self.block = Mixing(time_len=time_len,input_dim=input_dim)
        self.n_block = n_block
        self.flatten = nn.Flatten()
        self.pred = nn.Linear(input_dim*time_len,1)

    def forward(self,x):
        for _ in range(self.n_block):
            x = self.block(x)
        x = self.flatten(x)
        x = self.pred(x)
        return x


if __name__ == "__main__":
    data = torch.randn(128,15,32)
    model = TSMixer(input_dim=32,time_len=15,n_block=3)
    out = model(data)
    print(out.shape)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total params: {total_params}')

