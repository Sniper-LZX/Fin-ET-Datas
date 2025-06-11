#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/12/10 14:03
# Author  : SJ_Sniper
# File    : getRSI.py
# Description : 计算RSI并加入数据属性中
import json
import os
import re
import pandas as pd
from datetime import datetime, timedelta


# 计算RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    rsi = 100 - (100 / (1 + RS))
    return rsi


def get_rsi(file_name, start_date, end_date):
    file_path_xlsx = os.path.join("Shares_dataframe", file_name + ".xlsx")
    # 读取表格文件
    df = pd.read_excel(file_path_xlsx)  # 替换为你的表格文件路径

    # 将日期列转换为日期时间格式
    df['日期'] = pd.to_datetime(df['日期'])

    # 筛选指定日期范围内的行
    filtered_df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]

    # 计算RSI
    rsi = calculate_rsi(filtered_df['收盘'])
    print(rsi)
    # 打印计算结果
    return rsi.iloc[-1]


all_data = json.load(open("提取数据.json", "r", encoding="UTF-8"))
print("原来项目数量: ", len(all_data))

new_data = []

for item in all_data:
    # 原始数据信息
    type_times = item["Type_Time"]
    code = item["Report"][:6]
    id = item["Id"]
    rsi_list = []
    # 事件类型（时间范围）去重列表
    new_type_times = []
    # 开始遍历数据
    for i, one_type_time in enumerate(type_times):
        if one_type_time in new_type_times:
            continue
        # re匹配事件类型
        event_type = re.findall(r'^(.*?)[（(]', one_type_time)[0]
        # re匹配时间范围
        time_range = re.findall('(（.*?）)', one_type_time)[0][1:-1]

        # 开始日期和结束日期，并转换为日期类型
        begin_time = time_range.split('-')[0].replace("年", '-').replace("月", '-').replace("日", "")
        end_time = time_range.split('-')[1].replace("年", '-').replace("月", '-').replace("日", "")
        begin_datetime = datetime.strptime(time_range.split('-')[0], "%Y年%m月%d日")
        end_datetime = datetime.strptime(time_range.split('-')[1], "%Y年%m月%d日")

        # 扩展日期监控范围
        if begin_time == end_time:
            end_datetime += timedelta(days=61)
        elif end_datetime - begin_datetime < timedelta(days=25):
            end_datetime += timedelta(days=31)

        begin_time = begin_datetime.strftime("%Y年%m月%d日")
        end_time = end_datetime.strftime("%Y年%m月%d日")

        time_string = event_type + '（' + begin_time + '-' + end_time + '）'

        if time_string not in new_type_times:
            new_type_times.append(time_string)
        else:
            new_type_times.append(one_type_time)
        begin_time = begin_time.replace("年", '-').replace("月", '-').replace("日", "")
        end_time = end_time.replace("年", '-').replace("月", '-').replace("日", "")

        try:
            # TODO 计算 RSI 并放入数据中
            rsi = get_rsi(code, begin_time, end_time)
            rsi_list.append(rsi)
            item["RSI"] = rsi_list
            new_data.append(item)
        except Exception:
            print("error: ", id)
            rsi_list.append("error")
    new_data.append(item)

# with open("最终数据.json", "w", encoding="utf-8") as new_file:
#     json.dump(new_data, new_file, indent=4, ensure_ascii=False)
#     new_file.close()
