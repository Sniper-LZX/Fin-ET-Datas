from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def sentence_trans(txt_file_path, similar_file_path):
    # 模型路径
    model = SentenceTransformer('all-mpnet-base-v2')
    tag = ['股本变动与股东情况', '董事会讨论与分析报告', '重要事项', '讨论与分析']

    file_data2 = open(similar_file_path, 'w', encoding='utf-8')
    # 文本路径
    with open(txt_file_path, 'r', encoding='utf-8') as file_data:
        data = file_data.read()
        contents_list = [y for y in data.split('-----##-----') if y and y != '\n']

        for i, content in enumerate(contents_list):
            if content in tag:
                if i != 0:
                    file_data2.write('\n')
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
