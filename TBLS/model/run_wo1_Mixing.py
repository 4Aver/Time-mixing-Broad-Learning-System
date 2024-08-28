import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from 对比实验.utils import get_all_result
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from 研究生课题.集成学习与宽度学习.MBLS.model.wo1_Mixing import wo1_Mixing


warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 使用gpu
batch_size = 128
criterion = nn.MSELoss()
torch.manual_seed(21)       # 为CPU中设置种子，生成随机数。一旦固定种子，后面依次生成的随机数其实都是固定的。后面在生成loader时打乱顺序每次运行的结果一样
np.random.seed(21)

start_time = time.time()


def get_data(x, y, time_len,data_len,split_size):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)

    max_len = x.shape[0] - time_len + 1
    list_x, list_y = [], []
    for i in range(max_len):
        list_x.append(x[i:i+time_len,:])
        list_y.append(y[i+time_len-1,:])
    x, y = np.array(list_x), np.array(list_y)

    data = x[:data_len, :]
    test_data = x[data_len:, :]

    label = y[:data_len, :]
    test_label = y[data_len:, :]

    train_X, valid_X, train_y, valid_y= train_test_split(data, label, train_size=split_size, random_state=42)

    # 将数据转化为Tensor
    # 训练集
    train_seq = torch.from_numpy(np.array(train_X)).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)
    # 验证集
    valid_seq = torch.from_numpy(np.array(valid_X)).type(torch.FloatTensor)
    valid_label = torch.from_numpy(np.array(valid_y)).type(torch.FloatTensor)
    # 测试集
    test_seq = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_label = torch.from_numpy(np.array(test_label)).type(torch.FloatTensor)
    print('----------train--------')
    print(f'train X shape:{train_seq.shape}')
    print(f'train y shape:{train_label.shape}')

    print('----------val--------')
    print(f'val X shape:{valid_seq.shape}')
    print(f'val y shape:{valid_label.shape}')

    print('----------test--------')
    print(f'test X shape:{test_seq.shape}')
    print(f'test y shape:{test_label.shape}')

    train_dataset = TensorDataset(train_seq,train_label)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

    val_dataset = TensorDataset(valid_seq,valid_label)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)

    test_dataset = TensorDataset(test_seq,test_label)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    return train_dataloader,val_dataloader,test_dataloader,train_seq,train_label,valid_seq,valid_label,test_seq,test_label


def train(model, optimizer, scheduler, epoch, train_dataloader, train_seq):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch_index,batch_data in enumerate(train_dataloader):
        data,targets = batch_data       # torch.Size([128, 3, 5]) torch.Size([128, 1])

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 防止梯度爆炸或梯度消失问题
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_seq) / batch_size/5 )
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch_index, len(train_seq) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()
    return model


def evaluate(eval_model, dataloader, data_source_x):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for batch_index,dataset in enumerate(dataloader):
            data, targets = dataset

            output = eval_model(data)
            total_loss +=  len(data)*criterion(output, targets).cpu().item()
            result = torch.cat((result, output.cpu()),0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, targets.cpu()), 0)
    return total_loss / len(data_source_x),result,truth


def read_npy(type_name,epochs,choice_MLP,input_dim,time_len,map_fea_num, map_num, enh_fea_num, enh_num,weight_decay,lr = 0.001,hidden_dim1=256):
    # 切分数据
    if type_name=='sun_spot':
        data_x = pd.read_excel(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\sun_spot-x.xlsx').iloc[:,1:]
        data_y = pd.read_excel(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\sun_spot-y.xlsx').iloc[:,1:]
        data_len,split_size = 2700, 0.8
    else:
        data_x = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_x.npy'.format(type_name),allow_pickle=True)
        data_y = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_y.npy'.format(type_name),allow_pickle=True)
        data_len,split_size = 8500, 0.8

    print('data x and y shape:', data_x.shape, data_y.shape)

    train_dataloader,val_dataloader,test_dataloader,\
    train_seq,train_label,\
    valid_seq,valid_label,\
    test_seq,test_label = get_data(data_x, data_y,time_len=time_len,data_len=data_len,split_size=split_size)

    # 定义模型
    model = wo1_Mixing(choice_MLP=choice_MLP,input_dim=input_dim,time_len=time_len,
                 map_fea_num=map_fea_num, map_num=map_num, enh_fea_num=enh_fea_num, enh_num=enh_num,hidden_dim1=hidden_dim1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.98)       # 学习率调度器对象,调度器会在每个10个epoch后将学习率乘以gamma。
    best_test_loss = 1000
    best_val_output, best_val_target,best_test_output, best_test_target = 0, 0, 0, 0
    best_model = None

    # 训练模型
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model = train(model, optimizer, scheduler, epoch,train_dataloader, train_seq)

        train_loss, train_output, train_target = evaluate(train_model, train_dataloader, train_seq)
        valid_loss, valid_output, valid_target = evaluate(train_model, val_dataloader, valid_seq)
        test_loss, test_output, test_target = evaluate(train_model, test_dataloader, test_seq)

        if test_loss<best_test_loss:
            best_test_loss = test_loss
            best_test_output = test_output
            best_test_target = test_target
            best_model = train_model

        print('-' * 89)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.10f} | train loss {:.10f}| test loss {:.10f} '.format(
                epoch, (time.time() - epoch_start_time),
                valid_loss, train_loss, test_loss))
        scheduler.step()        # 学习率调度器对象

    mse, rmse, mae, mape, r2 = get_all_result(best_test_target, best_test_output)
    data_out = pd.DataFrame({'true':[i[0] for i in best_test_target.tolist()],
                             'pred':[i[0] for i in best_test_output.tolist()]},
                            index=range(len(best_test_output)))
    data_out.to_csv(r'C:\Users\86178\Desktop\课题\论文\MBLS\最终实验\{}\{}_wo_Mixing.csv'.format(type_name,type_name))
    end_time = time.time()
    print('计算所花费的时间为：', end_time - start_time)

    plt.plot(best_test_target, label='真实值')
    plt.plot(best_test_output, label='预测值')
    plt.legend()
    plt.show()
    return mse,r2


def choice(type_name,input_dim,map_fea_num, map_num, enh_fea_num, enh_num,hidden_dim1=64):
    list_time = []
    for i in [0,1,2,3,4]:
        start_time = time.time()
        read_npy(type_name,input_dim=input_dim,choice_MLP=True,time_len=5,
                     map_fea_num=map_fea_num, map_num=map_num, enh_fea_num=enh_fea_num, enh_num=enh_num,weight_decay=0.002,epochs=300)
        end_time = time.time()
        cost_time = end_time - start_time
        print('计算所花费的时间为：', end_time - start_time)
        list_time.append(cost_time)

choice('lorenz',input_dim=15,map_fea_num=32, map_num=3, enh_fea_num=10, enh_num=2)
# choice('rossler',input_dim=13,map_fea_num=24, map_num=5, enh_fea_num=1, enh_num=5)
# choice('power',input_dim=32,map_fea_num=32, map_num=3, enh_fea_num=10, enh_num=2)