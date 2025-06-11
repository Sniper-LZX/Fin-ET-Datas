# -*- coding: utf-8 -*-
from zhipuai import ZhipuAI
from Data_Deal.FilesScan import Files_Scan
from LTP.Sentence_Parser_Inter import LtpParser
import os, re, json
import time
import pyttsx3
from alive_progress import alive_bar


# 语音播报
def bobao(say_txt, volume):
    engine = pyttsx3.init()  # 创建engine并初始化
    engine.setProperty('rate', 150)
    engine.setProperty('volume', volume)
    engine.say(say_txt)
    engine.runAndWait()  # 等待语音播报完毕


path = 'Data/Test_batch'
# 文件列表
# txt_file_paths = Files_Scan(path, ['年报', '相似合并结果'], ['.json']).files_list
txt_file_paths = Files_Scan(path, ['年报', 'Result']).files_list
# 模型 api相关参数配置
client = ZhipuAI(api_key="f892a0c333042b4d83e65008c8be5ca3.ey96VlPjReoUD6Zy")
# 板块主题
sections = ['股本变动与股东情况', '董事会讨论与分析报告', '重要事项', '讨论与分析']


def inquiry(report_yearly):
    # 问句模板
    question_design = """
            你现在是一个自然语言处理的专家，需要你处理来自公司年报 {} 的文本。
            你需要：
                1、输入下面的段落文本。
                2、输出事件类型和时间范围。
            仅仅使用你对于整个段落的理解。
            以下是两个例子：
                输入：公司2011年1-11月实现销售收入5.2亿元，净利润6411.37万元。
                输出：公司盈利（2011年1月1月-2011年11月30日）
                输入：公司已于2011年12月份，将持有的该公司80%的股权转让，不再持有该公司股权。
                输出：股份转让（2011年12月1日-2011年12月31日）
            请注意，你仅仅需要显示两句话，请严格遵循如下格式：
                输出：输出内容1
                理由：理由1
                输出：输出内容2
                理由：理由2
                ......
            以下为要求的条件：
            （1）理由为判断事件类型以及时间范围的理由，而且时间范围中起始时间不能超过终止时间。
            （2）若时间只到年则以整年为范围，若时间只到月，则以整月为范围
            （3）若事件范围为报告期，则以一年为期限
            （4）你所标注的事件类型应与“公司盈利”“公司亏损”“股份转让”等事件类型处于同一级别
            请注意，你必须一个输出对应一个理由，不需要最后的理由说明。
            文本如下：
            
            """.format(report_yearly)
    return question_design

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
    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        top_p=0.7,
        temperature=0.5,
        max_tokens=8000,
        stream=True,
    )
    return response


def is_exist_nt(paragraph):
    """ 判断此句话中是否存在词性为 nt 即‘时间名词 temporal noun’ """
    ltp = LtpParser(r'../LTP/ltp_data')
    word_attribution = ltp.parser_main(paragraph)
    if 'nt' in word_attribution:
        return True
    return False


def json_fill(json_info, index, name, tag, para, type_time, reason):
    json_info['Id'] = index
    json_info['Report'] = name
    json_info['Tag'] = tag
    json_info['Input'] = para.replace('\n', '')
    json_info['Type_Time'] = type_time
    json_info['Reason'] = reason
    return json_info


if __name__ == '__main__':
    # 文本开始处理的时间戳
    file_t_start = time.time()
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
                all_data = []
                # 抽取不同板块（讨论与分析，董事会报告，重要事项）
                section = ''
                for content in contents_list:
                    if content in sections:
                        section = content
                        continue
                    paragraph_index = 1
                    print("目前抽取板块: %s" % section)
                    # 遍历所有段落
                    paragraphs = [y for y in content.split('\n') if y and y != '\n' and y != '\n\n']
                    with alive_bar(len(paragraphs), length=20, bar="bubbles", spinner="crab", force_tty=True) as bar:
                        for paragraph in paragraphs:
                            try:
                                json_content = {}
                                if not is_exist_nt(paragraph):
                                    print(' {} 号段落中不存在时间元素.'.format(paragraph_index))
                                    paragraph_index += 1
                                    bar()
                                    continue
                                if len(paragraph) <= 100:
                                    print(' {} 号段落中内容未超过100字.'.format(paragraph_index))
                                    paragraph_index += 1
                                    bar()
                                    continue
                                # 问句的模板加要询问的内容
                                question = inquiry(name[:-6]) + paragraph
                                response = get_answer(question)
                                # 结果
                                answer = ''
                                for trunk in response:
                                    answer += trunk.choices[0].delta.content
                                print(answer)
                                try:
                                    Type_Time = re.findall(r'输出：(.*)', answer)
                                    Reason = re.findall(r'理由：(.*)', answer)
                                except IndexError:
                                    print('问题句子:', paragraph)
                                    print(answer)
                                    bar()
                                    continue

                                json_content = json_fill(json_content, index, name[:-6], section, paragraph, Type_Time, Reason)
                                all_data.append(json_content)
                                paragraph_index += 1
                                index += 1
                                save_json.seek(0)
                                save_json.truncate()
                                json.dump(all_data, save_json, indent=4, ensure_ascii=False)
                                save_json.flush()
                            except Exception as e:
                                print('出现错误', e)
                                bar()
                                continue
                            bar()
            save_json.close()
            print('写入完成！')
        file_data.close()
        name_slash = name.split('-')
        bobao("{}的{}年年报抽取完毕".format(name_slash[0], name_slash[1]), 0.25)
    # 文本结束处理的时间戳
    file_t_end = time.time()

    # 消耗的时间
    inter = file_t_end - file_t_start
    minute, sec = divmod(inter, 60)
    hour, minute = divmod(minute, 60)
    period = "花费时间: {}小时 {}分钟 {:.2f}秒\n".format(hour, minute, sec)
    print(period)
    bobao("全部抽取完毕", 0.5)
