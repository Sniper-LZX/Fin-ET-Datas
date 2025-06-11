#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/3/30 9:37
# Author  : SJ_Sniper
# File    : TongyiControl.py
# Description :

# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
import re
from http import HTTPStatus
import dashscope

dashscope.api_key='sk-a449c5564c1544579d8ebb21f49bd7ac' #填入第一步获取的APIKEY

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


def call_with_messages(question):
    messages = [{'role': 'system', 'content': 'You are a natural language processing expert.'},
                {'role': 'user', 'content': question}]

    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        top_p=0.7,
        temperature=0.5,
        result_format='message',  # set the result to be "message" format.
    )
    print(response.output.choices[0].message.content)


if __name__ == '__main__':
    data = '''
焦炭M40转鼓指数81.78%，比上年提高0.07%；M10转鼓指数7.10%，比上年提高0.06%；焦炭灰分11.87%，比上年下降1.08%；焦炭硫分0.70%，比上年下降0.04%。转炉钢铁料消耗（一、二炼钢综合）1098.42千克/吨，比上年提高7.18千克/吨，其中吨钢废钢铁消耗147.62千克/吨，比上年提高28.16千克/吨。    '''
    question = question_design + data
    call_with_messages(question)


