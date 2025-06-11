#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/12/26 0:42
# Author  : SJ_Sniper
# File    : 统计.py
# Description :

import json, re

all_data = json.load(open("../../LLMs/Data_Deal/integrate/最终数据.json", "r", encoding="UTF-8"))

type_num = {}
report_num = []
for item in all_data:
    type_times = item["Type_Time"]
    report = item["Report"][:6]
    if report not in report_num:
        report_num.append(report)
    for one_type_time in type_times:
        event_type = re.findall(r'^(.*?)[（(]', one_type_time)[0]
        if event_type not in type_num:
            type_num[event_type] = 1
        else:
            type_num[event_type] += 1
print(len(report_num))
type_num = dict(sorted(type_num.items(), key=lambda item: item[1], reverse=False))
print(len(type_num), type_num)

