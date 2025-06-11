#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/8/1 13:32
# Author  : SJ_Sniper
# File    : SharesVisible.py
# Description :
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
matplotlib.rcParams['font.size'] = 24
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
# 读取Excel文件中的数据
file_path = '600009part.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel表格中有一个名为'Change'的列表示涨跌幅
# 绘制涨跌幅图表
plt.figure(figsize=(20, 12))
plt.plot(data['日期'], data['涨跌幅'], marker='o', linestyle='-', color='b')
plt.title('股票代码600009', pad=20)
plt.xlabel('日期', labelpad=20)
plt.ylabel('涨跌幅 (%)', labelpad=20)
plt.grid(True)
plt.savefig("幅度.png", format='png', dpi=300)
plt.show()


