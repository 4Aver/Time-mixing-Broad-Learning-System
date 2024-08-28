import torch
import random
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


class wo1_Mixing(nn.Module):
    def __init__(self,choice_MLP,input_dim,time_len,map_fea_num, map_num, enh_fea_num, enh_num,hidden_dim1=256):
        super(wo1_Mixing, self).__init__()

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

            for _ in range(enh_num):
                enh_seq = nn.Sequential(
                    nn.Linear(map_fea_num * map_num, hidden_dim1),
                    nn.ReLU(),
                    # nn.Dropout(0.01),
                    nn.Linear(hidden_dim1, enh_fea_num),
                    # nn.Dropout(0.01),
                )
                self.model_list_enh.append(enh_seq)

        else:
            for _ in range(map_num):
                map_dense = nn.Sequential(
                    nn.Linear(input_dim, map_fea_num),
                    # nn.ReLU(),
                )
                self.model_list_map.append(map_dense)

            for _ in range(enh_num):
                enh_seq = nn.Sequential(
                    nn.Linear(map_fea_num * map_num, enh_fea_num),
                    nn.Tanh(),
                )
                self.model_list_enh.append(enh_seq)

        self.dense_predict = nn.Linear(map_fea_num*map_num+enh_fea_num*enh_num+input_dim,1)
        self.dense_time = nn.Linear(time_len,1)
        self.identity = nn.Identity()

        self.bn = nn.BatchNorm2d(time_len,map_fea_num*map_num)

    def forward(self,x):
        input_x = self.identity(x)

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
    data = torch.randn(128,5,59)
    model = wo1_Mixing(choice_MLP=True,input_dim=59,time_len=5,map_fea_num=6, map_num=3, enh_fea_num=300, enh_num=1)
    out = model(data)
    print(out.shape)