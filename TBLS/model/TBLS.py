import torch
import torch.nn as nn


class Mixing(nn.Module):
    def __init__(self,time_len,input_dim):
        super(Mixing, self).__init__()
        self.identity = nn.Identity()

        # Time Mixing
        self.fnn_t = nn.Linear(time_len,1)
        self.relu_t = nn.ReLU()
        self.drop_t = nn.Dropout(p=0.1)
        # self.bn = nn.BatchNorm2d(time_len,input_dim)

    def forward(self,x):
        input_x = self.identity(x)      # [Batch, Input Length, Channel]
        # x = self.bn(x.unsqueeze(-1)).squeeze(-1)
        x = x.permute(0,2,1)       # [Batch, Channel, Input Length]
        x = self.fnn_t(x)
        x = self.relu_t(x)
        x = self.drop_t(x)
        x = x.permute(0,2,1)       # [Batch, Input Length, Channel]
        res = x + input_x
        return res


class MBLS(nn.Module):
    def __init__(self,choice_MLP,input_dim,time_len,n_block,map_fea_num, map_num, enh_fea_num, enh_num,hidden_dim1=256):
        super(MBLS, self).__init__()
        self.block = Mixing(time_len=time_len,input_dim=input_dim)
        self.n_block = n_block

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

        # self.bn = nn.BatchNorm2d(time_len,map_fea_num*map_num)

    def forward(self,x):
        for _ in range(self.n_block):
            x = self.block(x)
        input_x = self.identity(x)

        combine_features = torch.Tensor(0)
        for i in range(self.map_num):
            map_dense = self.model_list_map[i]
            map_feature = map_dense(x)
            combine_features = torch.cat((combine_features,map_feature),dim=2)

        # combine_features = self.bn(combine_features.unsqueeze(-1)).squeeze(-1)
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
    data = torch.randn(128,15,13)       # B,L,D
    model = MBLS(choice_MLP=True,input_dim=13,time_len=15,n_block=3,map_fea_num=32, map_num=3, enh_fea_num=10, enh_num=2)
    out = model(data)
    print(out.shape)

    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f'Total params: {total_params}')



