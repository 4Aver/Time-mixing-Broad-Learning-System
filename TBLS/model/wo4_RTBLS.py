'''
消融实验：测试随机权重时的MBLS预测效果
'''
import torch
import random
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(21)
def tansig(x):
    return (2/(1+torch.exp(-2*x)))-1


class Mixing(nn.Module):
    def __init__(self,time_len):
        super(Mixing, self).__init__()
        self.identity = nn.Identity()

        # Time Mixing
        self.fnn_t = nn.Linear(time_len,1)
        self.relu_t = nn.ReLU()
        self.drop_t = nn.Dropout(p=0.1)
        self.ln = nn.BatchNorm1d(time_len)


    def forward(self,x):
        input_x = self.identity(x)      # [Batch, Input Length, Channel]
        # x = self.ln(x)
        x = x.permute(0,2,1)       # [Batch, Channel, Input Length]
        x = self.fnn_t(x)
        x = self.relu_t(x)
        x = self.drop_t(x)
        x = x.permute(0,2,1)       # [Batch, Input Length, Channel]
        res = x + input_x
        return res


class wo4_RTBLS(nn.Module):
    def __init__(self,choice_MLP,input_dim,time_len,n_block,map_fea_num, map_num, enh_fea_num, enh_num):
        super(wo4_RTBLS, self).__init__()
        self.block = Mixing(time_len=time_len)
        self.n_block = n_block

        self.time_len = time_len
        self.map_fea_num = map_fea_num
        self.enh_fea_num = enh_fea_num
        self.map_num = map_num
        self.enh_num = enh_num
        self.model_list_map = nn.ModuleList()
        self.model_list_enh = nn.ModuleList()

        if choice_MLP:
            for _ in range(map_num):
                # map_dense = nn.Linear(input_dim,map_fea_num)
                map_dense = nn.Sequential(
                    nn.Linear(input_dim, map_fea_num),
                    nn.Sigmoid(),
                )
                self.model_list_map.append(map_dense)

        else:
            for _ in range(map_num):
                map_dense = nn.Sequential(
                    nn.Linear(input_dim, map_fea_num),
                )
                self.model_list_map.append(map_dense)

        self.tanh = nn.Tanh()
        self.dense_predict = nn.Linear(map_fea_num*map_num+enh_fea_num*enh_num,1)
        self.dense_time = nn.Linear(time_len,1)
        self.identity = nn.Identity()

    def forward(self,x):
        input_x = self.identity(x)
        for _ in range(self.n_block):
            x = self.block(x)
        combine_features = torch.Tensor(0)

        for i in range(self.map_num):
            map_dense = self.model_list_map[i]
            map_feature = map_dense(x)
            combine_features = torch.cat((combine_features,map_feature),dim=2)
        combine_features = self.dense_time(combine_features.permute(0,2,1)).squeeze()

        map_features = combine_features
        for i in range(self.enh_num):
            torch.manual_seed(21)
            enh_fea_weight = 2 * torch.randn(self.map_num * self.map_fea_num + 1, self.enh_fea_num) - 1  # [5*6+1,41]

            H2 = torch.hstack([map_features, 0.1 * torch.ones([map_features.shape[0], 1])])  # [3000,31]
            T2 = torch.matmul(H2,enh_fea_weight)
            T2 = tansig(T2)
            combine_features = torch.hstack([combine_features, T2])  # [3000,71]

        out = self.dense_predict(combine_features)
        return out


if __name__ == "__main__":
    data = torch.randn(128,5,59)
    model = wo4_RTBLS(choice_MLP=True,input_dim=59,time_len=5,n_block=3,map_fea_num=6, map_num=3, enh_fea_num=300, enh_num=2)
    out = model(data)
    print(out.shape)
