#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/12/29 21:41
# Author  : SJ_Sniper
# File    : split_data.py
# Description : 分割数据集为(训练:测试 = 7:3)

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# 加载数据
data = pd.read_pickle('136权重.pkl')
print(len(data))

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


with open('train_data_136权重.pkl', 'wb') as f:
    print(len(train_data))
    pickle.dump(train_data, f)

with open('test_data_136权重.pkl', 'wb') as f:
    print(len(test_data))
    pickle.dump(test_data, f)


