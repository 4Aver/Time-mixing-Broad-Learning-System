from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
from 对比实验.utils import get_all_result
import warnings
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_npy(type_name,input_dim,time_len):
    start_time = time.time()
    data_x = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_x.npy'.format(type_name), allow_pickle=True)
    data_y = np.load(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_y.npy'.format(type_name), allow_pickle=True)
    data_len,split_size = 8500, 0.8

    print('data x and y shape:', data_x.shape, data_y.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(data_x)
    y = scaler.fit_transform(data_y)
    x = x[:,:input_dim]

    max_len = x.shape[0] - time_len + 1
    list_x, list_y = [], []
    for i in range(max_len):
      list_x.append(x[i:i + time_len, :])
      list_y.append(y[i + time_len - 1, :])
    x, y = np.array(list_x), np.array(list_y)

    data = x[:data_len, :]
    test_data = x[data_len:, :]

    label = y[:data_len, :]
    test_label = y[data_len:, :]

    train_X, valid_X, train_y, valid_y = train_test_split(data, label, train_size=0.8, random_state=0)
    print(train_X.shape,train_y.shape)

    # 构建模型
    model = Sequential()
    model.add(LSTM(128, input_shape=(time_len, input_dim),return_sequences=True,activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(16,return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    tf.keras.optimizers.Adam(learning_rate=0.0001)

    # 定义一个ModelCheckpoint回调函数
    checkpoint_path = "best_model_weights.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # 模型训练
    history = model.fit(train_X, train_y, epochs=300, batch_size=128, verbose=2,
                        validation_data=(test_data, test_label),
                        callbacks=[checkpoint])

    # 找到验证集上最小的损失值对应的epoch索引
    best_epoch = np.argmin(history.history['val_loss'])
    print(best_epoch)

    # 使用最好的模型进行预测
    best_model = model  # 假设最后一个模型是最好的
    # best_model.load_weights('best_model_weights.h5')  # 加载最好的模型权重
    test_predictions_best = best_model.predict(test_data)

    # 输出最好情况下的预测数据
    data_out_best = pd.DataFrame({'true': [i[0] for i in test_label.tolist()],
                                  'pred': [i[0] for i in test_predictions_best.tolist()]},
                                 index=range(len(test_predictions_best)))
    data_out_best.to_csv(r'C:\Users\86178\Desktop\课题\论文\TBLS\最终实验\{}\{}_LSTM.csv'.format(type_name, type_name))

    end_time = time.time()
    print('计算所花费的时间为：', end_time - start_time)

    get_all_result(test_label,test_predictions_best)
    plt.plot(test_predictions_best,label='pred')
    plt.plot(test_label,label='true')
    plt.legend()
    plt.show()


# read_npy('lorenz',input_dim=15,time_len=15)         # 128 16 16 1
read_npy('rossler',input_dim=13,time_len=15)


