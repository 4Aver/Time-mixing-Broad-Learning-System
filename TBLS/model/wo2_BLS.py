import torch
import random
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


class Mixing(nn.Module):
    def __init__(self,time_len,input_dim):
        super(Mixing, self).__init__()
        self.identity = nn.Identity()

        # Time Mixing
        self.fnn_t = nn.Linear(time_len,1)
        self.relu_t = nn.ReLU()
        self.drop_t = nn.Dropout(p=0.1)
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
        return res


class wo2_BLS(nn.Module):
    def __init__(self,time_len,input_dim,n_block=1):
        super(wo2_BLS, self).__init__()
        self.n_block = n_block
        self.time_mixing_block = Mixing(time_len,input_dim)

        self.dense_time = nn.Linear(time_len,1)
        self.dense_feature = nn.Linear(input_dim,1)

    def forward(self,x):
        for _ in range(self.n_block):
            x = self.time_mixing_block(x)
        x = self.dense_feature(x).squeeze()
        x = self.dense_time(x)
        return x


if __name__ == "__main__":
    data = torch.randn(128,5,59)
    model = wo2_BLS(time_len=5,input_dim=59)
    output = model(data)
    print(output.shape)