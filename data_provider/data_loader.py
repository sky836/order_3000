import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features
from utils.tools import StandardScaler


class data(Dataset):
    def __init__(self, root_path, data_path='data.csv', flag='train', interval=60, size=None):
        self.root_path = root_path
        self.data_path = data_path

        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]

        # 用seq_len个序列长度去预测pred_len个序列长度
        if size == None:
            self.seq_len = 168
            self.pred_len = 168
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        self.interval = interval
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df = df.iloc[:, 1:]
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        dates = df.iloc[:, 0].values.astype('datetime64')

        interval = self.interval
        df['group'] = np.arange(len(df)) // interval  # 创建分组索引
        df_avg = df.groupby('group').agg({'humidity': 'mean', 'temperature': 'mean'}).reset_index(drop=True)  # 对每组取平均值并重置索引
        dates = dates[::interval]

        # 按7：1：2划分训练集、验证集和测试集
        l = len(df_avg)
        num_train = int(l * 0.7)
        num_test = int(l * 0.2)
        num_vali = l - num_train - num_test
        # 设置训练集、验证集、测试集的边界范围，会有seq_len + pred_len的数据用不上
        border1s = [0, num_train-self.seq_len, l-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, l]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        x = df_avg.values

        train_data = x[border1s[0]:border2s[0]]
        self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
        print('mean:', train_data.mean())
        print('std:', train_data.std())
        # 对数据进行标准化
        x = self.scaler.transform(x)

        data = x[border1:border2]
        data_y = df_avg.values[border1:border2]
        print('l:', len(data))
        feature_list = [data]
        # 对时间进行编码，返回是一个编码后的矩阵，每一行对应一个时间，列为编码后的特征
        stamp = time_features(pd.to_datetime(dates[border1:border2]), freq='T')
        # 进行转置，每一行对应一个特征，列为对应的时间
        stamp = stamp.transpose(1, 0)

        feature_list.append(stamp)
        processed_data = np.concatenate(feature_list, axis=-1)

        time_stamp = {'date': dates[border1:border2]}
        df_stamp = pd.DataFrame(time_stamp)
        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        df_stamp['minutes'] = df_stamp.date.apply(lambda row: row.minute)
        data_stamp = df_stamp.drop(columns=['date']).values

        self.data_x = data
        self.data_y = data_y
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    d = data(root_path='../', data_path='continues_clean_data.csv', flag='valid', interval=60)
    data_loader = DataLoader(
        d,
        shuffle=False,
        drop_last=True,
        batch_size=1
    )
    print(len(data_loader))