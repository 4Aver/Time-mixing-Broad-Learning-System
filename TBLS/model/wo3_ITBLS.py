'''
消融实验：测试通道独立性时的MBLS效果
'''
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers


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


class Channel_Independent(nn.Module):
    def __init__(self,list_input_dim,time_len):
        super(Channel_Independent, self).__init__()
        self.list_input_dim = list_input_dim
        self.time_len = time_len

    def forward(self,x):
        index = 0
        combine_features = torch.Tensor(0)
        for dim in self.list_input_dim:
            input_x = x[:,:,index:index+dim]
            index += dim

            temporal_model = Mixing(input_dim=dim,time_len=self.time_len)
            out = temporal_model(input_x)       # 在这里可以尝试卷积，将多个通道进行合并
            combine_features = torch.concatenate((combine_features,out),dim=2)
        return combine_features


class wo3_ITBLS(nn.Module):
    def __init__(self,list_input_dim,choice_MLP,time_len,map_fea_num, map_num, enh_fea_num, enh_num,hidden_dim):
        super(wo3_ITBLS, self).__init__()
        self.channel_independent = Channel_Independent(list_input_dim=list_input_dim,time_len=time_len)
        self.dependent_time = Mixing(time_len=time_len,input_dim=sum(list_input_dim))

        self.map_num = map_num
        self.enh_num = enh_num
        self.model_list_map = nn.ModuleList()
        self.model_list_enh = nn.ModuleList()

        self.bn_time = nn.BatchNorm2d(time_len,sum(list_input_dim))
        self.bn = nn.BatchNorm2d(time_len,map_fea_num*map_num)

        if choice_MLP:
            for _ in range(map_num):
                map_dense = nn.Sequential(
                    nn.Linear(sum(list_input_dim), map_fea_num),
                    nn.Sigmoid(),
                )
                self.model_list_map.append(map_dense)

            for _ in range(enh_num):
                enh_seq = nn.Sequential(
                    nn.Linear(map_fea_num * map_num, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, enh_fea_num),
                )
                self.model_list_enh.append(enh_seq)

        else:
            for _ in range(map_num):
                map_dense = nn.Sequential(
                    nn.Linear(sum(list_input_dim), map_fea_num),
                )
                self.model_list_map.append(map_dense)

            for _ in range(enh_num):
                enh_seq = nn.Sequential(
                    nn.Linear(map_fea_num * map_num, enh_fea_num),
                    nn.Tanh(),
                )
                self.model_list_enh.append(enh_seq)

        self.dense_predict = nn.Linear(map_fea_num*map_num+enh_fea_num*enh_num+sum(list_input_dim),1)
        self.dense_time = nn.Linear(time_len,1)
        self.identity = nn.Identity()

    def forward(self,x):
        input_x = self.identity(x)
        x = self.channel_independent(x)
        x = self.bn(x.unsqueeze(-1)).squeeze(-1)
        x = self.dependent_time(x)

        combine_features = torch.Tensor(0)
        for i in range(self.map_num):
            map_dense = self.model_list_map[i]
            map_feature = map_dense(x)
            combine_features = torch.cat((combine_features,map_feature),dim=2)

        combine_features = self.bn(combine_features.unsqueeze(-1)).squeeze(-1)
        map_features = combine_features
        for i in range(self.enh_num):
            enh_dense = self.model_list_enh[i]
            enh_feature = enh_dense(map_features)
            combine_features = torch.cat((combine_features, enh_feature), dim=2)

        residule_x = torch.concatenate((input_x,combine_features),dim=2)
        out = self.dense_predict(residule_x).squeeze()
        out = self.dense_time(out)
        return out


if __name__ == "__main__":
    data = torch.randn(128,5,14)
    # model = Channel_Independent([5,5,4],5)
    model = wo3_ITBLS(list_input_dim=[5,5,4],choice_MLP=True,time_len=5,map_fea_num=6, map_num=5, enh_fea_num=41, enh_num=1)
    out = model(data)
    print(out.shape)
