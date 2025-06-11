#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/7/27 13:55
# Author  : SJ_Sniper
# File    : SharesGet.py
# Description :
import akshare as ak
import os


def get_subfile_names(folder_path):
    # 获取文件夹中的所有文件和文件夹
    all_items = os.listdir(folder_path)
    return all_items


# 示例用法
folder_path = '../LLMs/Data_Deal/shanghaiYEARLY(10-17)'
entre_codes = get_subfile_names(folder_path)
print(entre_codes)
for code in entre_codes:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date="20100101",
        end_date="20200101",
        adjust="hfq"
    )
    xlsx_name = os.path.join('Shares_dataframe', code + '.xlsx')
    stock_zh_a_hist_df.to_excel(xlsx_name, index=False)
