#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/12/17 7:30
# Author  : SJ_Sniper
# File    : load_data.py
# Description : 构建叙事型多项选择题

import pickle
import re, json, random, copy
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import warnings
from transformers import logging
logging.set_verbosity_error()

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('../FinBERT_L-12_H-768_A-12_pytorch')
model = BertModel.from_pretrained('../FinBERT_L-12_H-768_A-12_pytorch')


def bert_cos(Text_1, Text_2):
    # 对单词进行编码
    tokens_a = tokenizer(Text_1, return_tensors='pt')
    tokens_b = tokenizer(Text_2, return_tensors='pt')

    # 获取单词的嵌入
    with torch.no_grad():
        embeddings_a = model(**tokens_a).last_hidden_state.mean(dim=1)
        embeddings_b = model(**tokens_b).last_hidden_state.mean(dim=1)

    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(embeddings_a, embeddings_b)
    cosine_similarity.item()
    # print(cosine_similarity.item())
    # print("{}和{}的相似度为{}".format(Text_1, Text_2, cosine_similarity.item()))
    return cosine_similarity.item()


def calculate_days(intervals):
    date1_obj = datetime.strptime(intervals[0], "%Y-%m-%d")
    date2_obj = datetime.strptime(intervals[1], "%Y-%m-%d")
    total_days = (date2_obj - date1_obj).days
    return total_days


def inter_and_union(inter1, inter2):
    # inter1 = ("2010-3-3", "2010-5-5")
    # inter2 = ("2010-4-4", "2010-6-6")
    inter_sec = (max(inter1[0], inter2[0]), min(inter1[1], inter2[1]))
    union_sec = (min(inter1[0], inter2[0]), max(inter1[1], inter2[1]))

    if datetime.strptime(inter_sec[0], "%Y-%m-%d") <= datetime.strptime(inter_sec[1], "%Y-%m-%d"):
        inter_days, union_days = calculate_days(inter_sec), calculate_days(union_sec)
        # print("交集天数：", inter_days)
        # print("并集天数：", union_days)
        return inter_days / union_days
    else:
        return 0.00


def calcu_weight(record_one, record_two):
    # TODO 两个事件项的序号
    id_one, id_two = int(record_one), int(record_two)
    type_one, type_two = int(10*(record_one - id_one)), int(10*(record_two - id_two))

    item_1, item_2 = all_data[id_one-1], all_data[id_two-1]
    rsi_1, rsi_2 = item_1["RSI"], item_2["RSI"]
    type_times_1 = item_1["Type_Time"][type_one - 1]
    type_times_2 = item_2["Type_Time"][type_two - 1]
    event_type_1 = re.findall(r'^(.*?)[（(]', type_times_1)[0]
    event_type_2 = re.findall(r'^(.*?)[（(]', type_times_2)[0]
    time_range_1 = re.findall('（(.*?)）', type_times_1)[0].split("-")
    time_range_2 = re.findall('（(.*?)）', type_times_2)[0].split("-")
    time_range_1 = tuple([datetime.strptime(x, "%Y年%m月%d日").strftime("%Y-%m-%d") for x in time_range_1])
    time_range_2 = tuple([datetime.strptime(x, "%Y年%m月%d日").strftime("%Y-%m-%d") for x in time_range_2])

    score_similar = bert_cos(event_type_1, event_type_2)
    score_time = inter_and_union(time_range_1, time_range_2)

    # TODO 为保持定向传播，时间上早的事件将指向晚的事件
    if time_range_1[0] > time_range_2[0]:
        return 0.6 * abs(rsi_1[type_one - 1] - rsi_2[type_two - 1]) / 100 + 0.3 * score_similar + 0.1 * score_time
    # return 0.3 * score_similar + 0.2 * score_time
    return 0.3 * score_similar


all_data = json.load(open("../../Data.json", "r", encoding="utf-8"))

context_rsi = []
candidates_rsi = []

report_all = {}
last_report = ""

for item in all_data:
    index = item["Id"]
    report = item["Report"][:6]
    report_time = item["Report"][:11]
    type_times = item["Type_Time"]
    if report not in report_all:
        if report not in report_all:
            report_all[report] = {}
            report_all[report][report_time] = []
            for i in range(len(type_times)):
                record = index + 0.1 * (i + 1)
                report_all[report][report_time].append(record)
    else:
        if report_time not in report_all[report]:
            report_all[report][report_time] = []
        for i in range(len(type_times)):
            record = index + 0.1 * (i + 1)
            report_all[report][report_time].append(record)

# 统计所有的索引（年报id.事件类型序号）
all_index = []
left, right = 1, 0
for report, report_sub in report_all.items():
    for report_time, index_list in report_sub.items():
        all_index += index_list
        right += len(index_list)
    report_all[report]["index_range"] = (left, right)
    left = right + 1

# print(report_all)
# print(all_index)

# TODO 训练数据集
train = []
ith = 1
punctuation = r'[^\w\s,，.。:：！？;；（）\n]'
for _ in range(2):
    for report, report_sub in report_all.items():
        left, right = report_all[report]["index_range"]
        numbers = list(range(1, left)) + list(range(right+1, len(all_index) + 1))
        # 随机选取节点
        selected_indexes = [all_index[x-1] for x in random.sample(numbers, 4)]
        # print(all_index[left-1], all_index[right-1], selected_indexes)

        for report_time, index_list in report_sub.items():
            data = {}
            # 文本
            context = []
            # 候选
            candidates = []
            # 事件图
            graph = []

            if len(index_list) <= 5 or report_time == "index_range":
                continue
            # 将正确事件加入到候选事件中
            print("数据集", _, report_time)
            edge = random.sample(index_list[:-1], 5)
            random_index = random.randint(0, len(candidates))
            selected_indexes.append(index_list[-1])

            # 构建文本列表（正文和候选）
            text_len = 512 // len(index_list)
            for index_context in edge:
                r_id = int(index_context)
                type_id = int(10 * (index_context - r_id))
                context.append(re.sub(punctuation, '', all_data[r_id - 1]["Input"])[:text_len])
            for index_candidate in selected_indexes:
                r_id = int(index_candidate)
                type_id = int(10 * (index_candidate - r_id))
                insert_inf = re.sub(punctuation, '', all_data[int(index_candidate) - 1]["Input"])[:text_len]
                candidates.insert(random_index, insert_inf)

            for select in selected_indexes:
                temp = edge[:]
                temp.append(select)
                # print("创建{}*{}的图--------".format(len(temp), len(temp)))
                matrix = [[0.00 for _ in range(len(temp))] for _ in range(len(temp))]
                for i in range(len(matrix)):
                    for j in range(len(matrix[0])):
                        if i == j:
                            continue
                        # 计算权重值
                        matrix[i][j] = calcu_weight(temp[i], temp[j])
                        # print("顶点{} -> 顶点{}: 权重为{}".format(i, j, matrix[i][j]))
                # print("矩阵为: ")
                # for m in matrix:
                #     print("[ ", end=" ")
                #     for n in m:
                #         print("{:.6f}".format(n), end=" ")
                #     print(" ]")
                graph.append([matrix])
            data["context"] = context
            data["candidates"] = candidates
            data["ith"] = ith
            data["graph"] = graph
            data["ans"] = random_index
            train.append(data)
            ith += 1
            selected_indexes = selected_indexes[:-1]
        #     if len(train) == 1:
        #         break
        # if len(train) == 1:
        #     break
file_name = "631权重.pkl".format(5)
with open(file_name, "wb") as f:
    pickle.dump(train, f)
