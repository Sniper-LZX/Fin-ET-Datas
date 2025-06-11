# -*- coding: utf-8 -*-
from Data_Deal.FilesScan import Files_Scan
from LTP.Sentence_Parser_Inter import LtpParser
import os, re, json
import time
import dashscope


path = 'Data/Test_batch'
# 文件列表
txt_file_paths = Files_Scan(path, ['年报', '相似合并结果'], ['.json']).files_list
# 模型 api相关参数配置
dashscope.api_key='sk-a449c5564c1544579d8ebb21f49bd7ac' #填入第一步获取的APIKEY
# 板块主题
tags = ['股本变动与股东情况', '董事会讨论与分析报告', '重要事项', '讨论与分析']
# 问句模板
question_design = """
        你现在是一个自然语言处理的专家，你需要：
            1、输入下面的段落文本。
            2、输出事件类型和时间范围。
        仅仅使用你对于整个段落的理解。
        以下是两个例子：
            输入：公司2011年1-11月实现销售收入5.2亿元，净利润6411.37万元。
            输出：公司盈利（2011年1月-2011年11月）
            输入：公司已于2011年12月份，将持有的该公司80%的股权转让，不再持有该公司股权。
            输出：股份转让（2011年12月-2011年12月）
        请注意，你仅仅需要显示两句话，请严格遵循如下格式：
            输出：输出内容
            理由：理由
            输出：
            理由：
            ......
        理由为判断的时间范围的理由，而且时间范围中起始时间不能超过终止时间。请注意，你必须第一个输出对应一个理由，不需要最后的理由说明。
        """


def get_path_information(path):
    # 除文件名外的文件目录
    catalog = os.path.dirname(path)
    # 文件全称
    file_basename = os.path.basename(path)
    # 文件名称和后缀
    file_name, file_extension = os.path.splitext(file_basename)
    return catalog, file_name, file_extension


def get_answer(question):
    # 询问模型的相关配置
    messages = [{'role': 'system', 'content': 'You are a natural language processing expert.'},
                {'role': 'user', 'content': question}]
    response = dashscope.Generation.call(
        model='qwen-max',
        messages=messages,
        top_p=0.7,
        temperature=0.5,
        result_format='message',  # set the result to be "message" format.
    )
    return response


def is_exist_nt(paragraph):
    """ 判断此句话中是否存在词性为 nt 即‘时间名词 temporal noun’ """
    ltp = LtpParser(r'../LTP/ltp_data')
    word_attribution = ltp.parser_main(paragraph)
    if 'nt' in word_attribution:
        return True
    return False


if __name__ == '__main__':
    for file_path in txt_file_paths:
        print(file_path)
        # 获得文件路径的目录，文件名称，文件后缀
        directory, name, extent = get_path_information(file_path)
        # 文本路径
        with open(file_path, 'r', encoding='utf-8') as file_data:
            save_json = os.path.join(directory, name[:-6] + '.json')
            index = 1
            with open(save_json, 'w', encoding='utf-8') as save_json:

                contents_list = [y for y in file_data.read().split('-----##-----')
                                 if y and y != '\n' and y != '\n\n']
                # 文本开始处理的时间戳
                file_t_start = time.time()
                all_data = []
                # 遍历所有段落
                tag = ''
                for content in contents_list:
                    if content in tags:
                        tag = content
                        continue
                    print('目前抽取模块:', tag)
                    paragraph_index = 1
                    for paragraph in [y for y in content.split('\n')
                                      if y and y != '\n' and y != '\n\n']:
                        try:
                            json_content = {}
                            print('目前抽取的段落的编号:', paragraph_index)
                            if not is_exist_nt(paragraph):
                                print('    ', paragraph_index, ' 段落中不存在时间元素。')
                                paragraph_index += 1
                                continue
                            if len(paragraph) <= 100:
                                print('    文字的长度没有超过100字。')
                                paragraph_index += 1
                                continue
                            # 问句的模板加要询问的内容
                            question = question_design + paragraph
                            response = get_answer(question)
                            # 结果
                            answer = response.output.choices[0].message.content
                            print(answer)

                            try:
                                Type_Time = re.findall(r'输出：(.*)', answer)
                                Reason = re.findall(r'理由：(.*)', answer)
                            except IndexError:
                                print('问题句子:', paragraph)
                                print(answer)
                                continue

                            json_content['Id'] = index
                            json_content['Report'] = name[:-6]
                            json_content['Tag'] = tag
                            json_content['Input'] = paragraph.replace('\n', '')
                            json_content['Type_Time'] = Type_Time
                            json_content['Reason'] = Reason
                            all_data.append(json_content)
                            paragraph_index += 1
                            index += 1
                            save_json.seek(0)
                            save_json.truncate()
                            json.dump(all_data, save_json, indent=4, ensure_ascii=False)
                            save_json.flush()
                        except Exception as e:
                            print('出现错误', e)
                            continue
                # 文本结束处理的时间戳
                file_t_end = time.time()
            save_json.close()
            print('写入完成！')
        file_data.close()

        # 消耗的时间
        inter = file_t_end - file_t_start
        minute, sec = divmod(inter, 60)
        hour, minute = divmod(minute, 60)
        period = "花费时间: {}小时 {}分钟 {:.2f}秒\n".format(hour, minute, sec)
        print(period)
