import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from 对比实验.utils import get_all_result
from sklearn.preprocessing import MinMaxScaler
import time

import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_npy(type_name):
    start_time = time.time()
    data_y = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_y.npy'.format(type_name),allow_pickle=True)
    data_len = 8400

    sc = MinMaxScaler(feature_range=(-1, 1))
    x = sc.fit_transform(data_y)

    features = np.array(x)[data_len:, :]
    target = np.array(x)[data_len:, :]

    length = features.shape[0]
    window = 100
    end_index = length - window
    list_X = []  # windows
    list_Y = []  # horizon
    index = 0
    while index < end_index:
        list_X.append(features[index:index + window])
        list_Y.append(target[index + window:index + window+1])
        index = index + 1
    X = np.array(list_X)
    Y = np.array(list_Y).reshape(len(list_Y), -1)
    print(X.shape, Y.shape)

    list_out = []
    for index in range(X.shape[0]):
        print('第{}次，共{}次'.format(index+1,X.shape[0]))
        features = X[index].reshape(X.shape[1], X.shape[2])
        target = Y[index]

        model = ARIMA(features, order=(1, 0, 1))
        model_fit = model.fit()
        out = model_fit.forecast(steps=1)

        list_out.append(out)

    get_all_result(Y, list_out)
    data_out = pd.DataFrame({'true':[i[0] for i in Y.tolist()],
                             'pred':[i[0] for i in list_out]},
                            index=range(len(list_out)))
    data_out.to_csv(r'C:\Users\86178\Desktop\课题\论文\TBLS\最终实验\{}\{}_ARIMA.csv'.format(type_name,type_name))
    end_time = time.time()
    print('计算所花费的时间为：', end_time - start_time)

    # plt.plot(Y, label='真实值')
    # plt.plot(list_out, label='预测值')
    # plt.legend()
    # plt.show()


# read_npy('lorenz')
read_npy('rossler')
# read_npy('power')
