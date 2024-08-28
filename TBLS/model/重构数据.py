import numpy as np


def read_npy(type_name):
    data = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\多维混沌时间序列\standard_data\{}_rec_dict.npy'.format(type_name),allow_pickle=True).tolist()
    list_name = list(data.keys())
    max_dim = max([data[name].shape[1] for name in list_name])

    target = data[list_name[0]][1:,0]
    # 填充零值，使所有输入维度一致
    for name in list_name:
        data[name] = data[name][:-1,:]
        num_sample, dim = data[name].shape
        if dim<5:
            zores = np.zeros((num_sample,max_dim-dim))
            data[name] = np.concatenate((data[name],zores),axis=1)
    np.save(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\多维重构(带空间信息)\x_{}_rec_fill_dict.npy'.format(type_name),data)
    np.save(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\多维重构(带空间信息)\y_{}_rec_fill_dict.npy'.format(type_name),target)

# read_npy('lorenz')
# read_npy('rossler')
# read_npy('sea_clutter')


# 查看重构信息
data = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\多维重构(带空间信息)\x_{}_rec_fill_dict.npy'.format('sea_clutter'),allow_pickle=True).tolist()
list_name = list(data.keys())
data_x = np.array([data[name] for name in list_name])
print(data_x.shape)