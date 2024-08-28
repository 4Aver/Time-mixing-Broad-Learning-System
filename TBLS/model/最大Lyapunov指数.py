import numpy as np
import pandas as pd
from nolds import lyap_e
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# data_lorenz = pd.read_csv(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\original\lorenz.csv').iloc[:10000,1:]
# data_rossler = pd.read_csv(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\original\rossler.csv').iloc[:10000,1:]
# data_sea_clutter = pd.read_excel(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\original\sea_clutter.xlsx').iloc[:10000,1:]
#

# def max_layapunov_seies(type_name,data):
#     # 计算Lyapunov指数
#     len = 1000
#     data_t_m = pd.read_excel(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\{}_tau_m(1).xlsx'.format(type_name))
#     list_tau = data_t_m.iloc[:, 1].tolist()
#     list_m = data_t_m.iloc[:, 2].tolist()
#
#     num_variable = data.shape[1]
#     list_lyapunov = []
#     for i in range(num_variable):
#         for j in range(0,10001-len,len):
#             x = data.iloc[j:j+len,i]
#             lyap_exp = lyap_e(x,emb_dim=list_m[i],matrix_dim=2)
#             print("Lyapunov指数:", lyap_exp)
#             list_lyapunov.append(lyap_exp[0])
#     lyapunov = np.array(list_lyapunov)
#     return lyapunov
#
# lyapunov_lorenz = max_layapunov_seies(type_name='lorenz',data=data_lorenz)
# lyapunov_rosser = max_layapunov_seies(type_name='rossler',data=data_rossler)
# lyapunov_sea_clutter = max_layapunov_seies(type_name='sea_clutter',data=data_sea_clutter)
#
# # lyapunov_all = np.concatenate((lyapunov_lorenz,lyapunov_rosser,lyapunov_sea_clutter))
# # print(lyapunov_all)
#
# # 使用gaussian_kde函数进行估计
# kde_lorenz = gaussian_kde(lyapunov_lorenz)
# kde_rossler = gaussian_kde(lyapunov_rosser)
# kde_sea_clutter = gaussian_kde(lyapunov_sea_clutter)
#
# # 绘制估计结果
# x_grid_1 = np.linspace(-0.2, 1, 100)
# x_grid_2 = np.linspace(-0.2, 1, 100)
# x_grid_3 = np.linspace(-0.2, 1, 100)
#
# plt.plot(x_grid_1, kde_lorenz(x_grid_1),label='lorenz')
# # plt.plot(x_grid_2, kde_rossler(x_grid_2),label='rossler')
# plt.plot(x_grid_3, kde_sea_clutter(x_grid_3),label='sea_clutter')
# plt.legend()
# plt.show()



data_sun_spot = pd.read_excel(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\data\sun spot\y-sun spot.xlsx',header=None).iloc[:,1]
lyap_exp = lyap_e(data_sun_spot,emb_dim=5,matrix_dim=2)
print("sun_spot Lyapunov指数:", lyap_exp)

data_rossler = pd.read_csv(r'C:\Users\86178\Desktop\课题\下载\混沌时间序列\重构\original\rossler.csv').iloc[:10000,1:]
lyap_exp = lyap_e(data_rossler.iloc[:,0],emb_dim=3,matrix_dim=2)
print("rossler Lyapunov指数:", lyap_exp)
