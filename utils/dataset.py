# -*-coding:utf-8-*-

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, feature_path, label_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))

    # 返回数据集大小
    def __len__(self):
        return len(self.feature_paths)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])
        feature_data = torch.from_numpy(feature_data)  # numpy转成张量
        label_data = torch.from_numpy(label_data)
        # feature_data.unsqueeze_(0)  # 增加一个维度128*128 =>1*128*128
        # label_data.unsqueeze_(0)
        return feature_data, label_data

if __name__ == "__main__":

    feature_path = "..\\data\\feature\\"
    label_path = "..\\data\\label\\"
    seismic_dataset = MyDataset(feature_path, label_path)
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,
                                               batch_size=32,
                                               shuffle=True)
    # Img = train_loader.numpy().astype(np.float32)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))
