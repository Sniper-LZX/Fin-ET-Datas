#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/12/11 7:41
# Author  : SJ_Sniper
# File    : 错误.py
# Description :

import json
import pandas as pd

e = json.load(open("错误.json", "r", encoding="utf-8"))

all_data2 = json.load(open("提取数据.json", "r", encoding="UTF-8"))
print("总共: ", len(all_data2))

x = []
y = []
for item in all_data2:
    rsi = item["RSI"]
    if pd.isna(rsi).all():
        x.append(item["Id"])
    else:
        if pd.isna(rsi).any():
            y.append(item["Id"])
e["nan"] = x
e["nan_any"] = y

print("全是nan", len(x))
print("存在nan", len(y))
print(y)
with open("错误4.json", "w", encoding="utf-8") as error_file:
    json.dump(e, error_file, indent=4, ensure_ascii=False)
    error_file.close()

