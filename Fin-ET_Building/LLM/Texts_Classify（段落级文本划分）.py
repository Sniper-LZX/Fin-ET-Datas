from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from Data_Deal.FilesScan import Files_Scan
import os


# 模型路径
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

# dir_path = 'Data/shanghaiYEARLY(10-17)deal'
dir_path = r'Data\shanghaiYEARLY(10-17)deal'
# txt_file_paths = Files_Scan(dir_path, ['年报'], ['相似合并结果']).files_list
txt_file_paths = Files_Scan(dir_path, ['相似合并结果', '.txt']).files_list

# 总文件
total = 0
# 未被处理的文件
un_tackle = 0
tag = ['股本变动与股东情况', '董事会讨论与分析报告', '重要事项', '讨论与分析']
for file_path in txt_file_paths:
    print(file_path)
    # 除文件名外的文件目录
    directory = os.path.dirname(file_path)
    # 文件全称
    file_basename = os.path.basename(file_path)
    # 文件名称和后缀
    file_name, file_extension = os.path.splitext(file_basename)
    # 结果保存文件路径
    save_dir = os.path.join('Data/句子相似融合实验',directory)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_txt = os.path.join(save_dir, file_name[:-6] + 'Result' + file_extension)
    file_data2 = open(save_txt, 'w', encoding='utf-8')
    # 文本路径
    with open(file_path, 'r', encoding='utf-8') as file_data:
        data = file_data.read()
        contents_list = [y for y in data.split('-----##-----') if y and y != '\n']

        for content in contents_list:
            if content in tag:
                content = '-----##-----' + content + '-----##-----'
                file_data2.write(content)
                file_data2.write('\n')
                continue
            # 分句
            data_list = [x for x in content.split('\n') if x and x != '\n' and x != '\n\n']
            # print(data_list)
            if data_list and len(data_list) > 1:
                # 文本相似分数集合
                similar_data = []

                for i in range(len(data_list)-1):
                    # 两个句子
                    sentences1 = data_list[i]
                    sentences2 = data_list[i+1]
                    # 句子嵌入
                    embeddings1 = model.encode(sentences1, show_progress_bar=True)
                    embeddings2 = model.encode(sentences2, show_progress_bar=True)
                    # 直接计算句子间相似度
                    cosine_scores = cos_sim(embeddings1, embeddings2).item()
                    similar_data.append(cosine_scores)
                scores_len = len(similar_data)

                # 滑动窗口，确保每次处理10个句子。
                # 若最后一组不够10个，则与倒数第二组一起处理，即最后处理的一组可能会超过10句话
                begin, end = 0, 9
                tags = []
                # 相似平均分数
                if scores_len > 9:
                    while end < scores_len - 9:
                        batch_sum, batch_scores_list = 0.00, []
                        for i in range(begin, end + 1):
                            batch_scores_list.append(similar_data[i])
                            batch_sum += similar_data[i]
                        batch_avg = batch_sum / 9.00
                        batch_tags = [1 if x >= batch_avg else 0 for x in batch_scores_list]
                        tags += batch_tags
                        begin = end + 1
                        end = begin + 8
                batch_sum, batch_scores_list = 0.00, []
                for j in range(begin, scores_len):
                    batch_scores_list.append(similar_data[j])
                    batch_sum += similar_data[j]
                # 计算平均余弦值
                batch_avg = batch_sum / float(len(batch_scores_list))
                # 大于平均的给1，小于的给0,
                batch_tags = [1 if x >= batch_avg else 0 for x in batch_scores_list]
                tags += batch_tags
                new_text = data_list[0]
                # 1代表合并
                for n in range(len(tags)):
                    if tags[n] == 1:
                        new_text += data_list[n + 1].replace('\n','')
                    else:
                        new_text = new_text + '\n\n' + data_list[n + 1].replace('\n','')
            else:
                if len(data_list) == 1:
                    new_text = data_list[0]
                else:
                    new_text = ''
            file_data2.write(new_text)
            file_data2.write('\n')
        file_data.close()
    file_data2.close()
    total += 1

# print("共有文件 {} 个".format(total))
# print("不存在对应文件的有 {} 个".format(un_tackle))

# """语言播放"""
# engine = pyttsx3.init()
# engine.say("执行完成")
# engine.runAndWait()
