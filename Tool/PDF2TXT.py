# coding:utf-8

import pdfplumber
import os


# PDF文档解析
# 解析的pdf数
global parser_pdf_count
# 无法解析pdf数
global cannot_parser_count
# 文件总数
global total

'''------------------保存年报的文件夹路径----------------------'''
# 企业年报
folder_path = 'test'

# 数字标题集
num_of_title = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
# 标点符号集
punc = ['；', '。', '，', '、']
# 目录中修饰页号的尾缀集合
# 两个'-'识别不同
mod = ['页', '', ']', '-', '‐', '—', ')', '】', '.']


# 目录符号集
def in_punc(x):
    cata_punc = ['„„„', '---', '......', '……', '______', '———']
    for y in cata_punc:
        if y in x:
            return True
    return False


# 跳过表格下方的注（部分格式）
def in_note(li, x):
    """
            使得跳过所有 “注：” 及其内容
    """
    if len(str(li[x])) >= 3:
        if li[x][2] == '（' and li[x][3] == '1':
            '''
                    注：（1）....
            '''
            while x < len(li) and not istitle(li, x):
                x += 1

        elif x < len(li) - 1:
            if li[x+1][0] == '（' and li[x+1][1] == '1':
                '''
                        注：......:
                        （1）.....
                        （2）.....
                '''
                r = 2
                x += 2
                while x < len(li):
                    if len(li[x]) > 2:
                        if li[x][0] == '（' and li[x][1] == str(r):
                            r += 1
                            x += 1
                        else:
                            x += 1
                    else:
                        x += 1
        elif li[x][-1] == '。':
            '''
                    注：....... 。
            '''
            x += 1
        elif '。' not in li[x] and len(li[x]) < 43:
            x += 1
    elif len(str(li[x])) == 2:
        x += 1
    return x


# 判断是否在表格中
def in_table(li, x, tli):
    """
                li[]是以行储存的一个页面列表
                tli[]是以行储存的表格列表
                    判断 li[x] 是否属于表格里的内容
    """
    if (li[x] in tli) or (li[x][:3] == "单位："):
        return True
    return False


# 判断是否是标题
def istitle(li, x):
    """
            li[]是以行储存的一个页面内容的列表
            判断 li[x] 是否是标题
    """
    tt = False  # 首先默认不是标题行
    if len(li[x]) >= 4:  # 如果一行长度大于1，则判断是否是标题行
        '''
            一、 二、 三、 四、 ...
            十一、 十二、 十三、 十四、 ...
            第一节/章 第二节/章 第三节/章 第四节/章 ...
            （一）（二）（三）（四） ...
            (一) (二) (三) (四) ...
            （1）（2）（3）（4）...
            (1) (2) (3) (4) ...
            1、 2、 3、 4、 5、 ...
            11、 12、 13、 14、 ...
            1.1 1.2 1.3 1.4 ...
            10.1 11.2 12.3 ...
        '''
        tt = (li[x][0] in num_of_title and li[x][1] == '、'
              and len(li[x]) < 40 and (li[x][-1] not in punc)) or \
            (li[x][0] in num_of_title and li[x][1] in num_of_title and li[x][2] == '、' 
             and len(li[x]) < 40 and (li[x][-1] not in punc)) or \
            (li[x][0] == '第' and li[x][1] in num_of_title and li[x][2] == '节') or \
            (li[x][0] == '（' and li[x][1] in num_of_title and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0] == '(' and li[x][1] in num_of_title and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0] == '（' and li[x][1].isdigit() and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0] == '(' and li[x][1].isdigit() and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0].isdigit() and li[x][1] == '、' and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0:2].isdigit() and li[x][2] == '、' and len(li[x]) < 38 and (li[x][-1] not in punc)) or \
            (li[x][0].isdigit() and li[x][1] == '.' and li[x][2].isdigit() and len(li[x]) < 38
             and '。' not in li[x] and '，' not in li[x] and '；' not in li[x]) or \
            (li[x][0:2].isdigit() and li[x][2] == '.' and li[x][3].isdigit() and len(li[x]) < 38
             and '。' not in li[x] and '，' not in li[x] and '；' not in li[x])

    return tt


# 获取目录中的页数
def get_page_num(catalog):
    """
            获取目录一行中小结的页号
    """
    page_num = 0
    page_str = ''
    while len(catalog) > 0 and catalog[-1] in mod:
        catalog = catalog[:-1]
    if catalog.isdigit():
        return int(catalog)
    j = len(catalog) - 1
    while j > 0:            # 从后向前遍历目录的一行
        if catalog[j].isdigit():                        # 判断字符是不是数字
            if '------' in catalog:
                while catalog[j].isdigit():
                    page_str = catalog[j] + page_str  # 获得页号
                    j -= 1
            else:
                while j > 0 and catalog[j].isdigit():
                    if (catalog[j] == '-' or catalog[j] == '—') and catalog[j - 1].isdigit():
                        '''
                        目录页号显示为 “xx-xx”
                        '''
                        page_str = ''
                        j -= 1
                    page_str = catalog[j] + page_str            # 获得页号
                    j -= 1

            if len(page_str) >= 4:
                '''..................6666'''
                page_str2 = ''
                num_page = len(page_str) // 4
                for u in range(num_page):
                    a = ''.join(set(list(page_str[u*4:u*4+4])))
                    page_str2 += a
                page_str = page_str2
            page_num = int(page_str)
            break
        else:
            if catalog[j] in mod:
                j -= 1
                continue
            else:
                return 0
    return page_num


# 获取文件夹内文件列表
def get_file_list(file_path, file_list):
    """
    -------------获取指定文件夹下的所有文件---------------
    输入文件夹或文件路径,返回文件列表 file_list, 内含文件路径
    """
    # 判断对象是否为文件
    if os.path.isfile(file_path):
        # 如果是文件就加入列表中
        file_list.append(file_path)
        # 返回路径中最后的文件名
        # print(os.path.basename(file_path))

    # 判断对象是否为文件夹目录
    elif os.path.isdir(file_path):
        # 遍历文件夹里面的文件或文件夹
        # 注意：os.listdir返回的是文件夹里的文件名字的列表，并不是完整路径
        for s in os.listdir(file_path):
            # 忽略某些文件或文件夹
            # if s == "xxx":
            #     continue
            # 递归查询是否是文件夹并继续
            new_path = os.path.join(file_path, s)
            get_file_list(new_path, file_list)


# 第二个入口，解析具体页面
def text_extra(chapter, add_page, begin_page, end_page, pdf, out_txt_name):
    """
                    数据处理，以及三元组抽取
    """
    if end_page == begin_page:
        end_page = begin_page + 1
    begin_page += add_page
    end_page += add_page
    chapter_flag = '-----##-----' + chapter + '-----##-----\n'
    # 页面正文内容写入txt文件
    with open(out_txt_name, 'a', encoding='utf-8') as ft:
        # 写入模块名
        ft.write(chapter_flag)
        # 用于记录标题之间的正文
        text = ''

        # 遍历所有指定的页面范围
        for j in range(begin_page - 1, end_page - 1):
            try:
                '''-------------------获取页面内容，每一行作为来列表的元素-------------------'''
                content = pdf.pages[j].extract_text()                                       # 获取页面内容
                content_list = content.replace(' ', '').replace(' ', '').split('\n')     # 去除空格并按行分割
                content_list = [x for x in content_list if x != '']                         # 去除列表里的空值

                if j == begin_page - 1:
                    for li, begin_line in enumerate(content_list):
                        if ("讨论" in begin_line and "分析" in begin_line) or "董事会报告" in begin_line or\
                                "重要事项" in begin_line or "重大事项" in begin_line or\
                                ("变动" in begin_line and "股东" in begin_line and "情况" in begin_line):
                            content_list = content_list[li:]

                # 存在表格中无法匹配的元素，尽量删除
                if content_list:
                    if len(content_list[0]) > 4:
                        if content_list[0][:4].isdigit() and content_list[0][4:] == "年年度报告":
                            content_list = content_list[1:]
                    if content_list[-1].isdigit():
                        # 数组最后一个是数字，即页号
                        content_list = content_list[:len(content_list)-1]

                '''-------------获取表格内容，每一行合并为字符串，作为来列表的元素-------------'''
                tables = pdf.pages[j].extract_tables()              # 解析表格
                table_list = list()
                for table in tables:
                    for o in range(len(table)):
                        table[o] = ''.join([x for x in table[o] if x]).replace(' ', '').replace(' ', '')
                        table_list.append(table[o])

                # 获取页面所有非标题的正文
                a = 0
                not_over = True
                while a < len(content_list):                        # 遍历列表里的所有行（一页）
                    ''' 属于表格内的信息，则跳过（格式简单的表格） '''
                    if in_table(content_list, a, table_list):
                        a += 1
                        continue
                    ''' 表格中遗漏掉的、无法匹配的内容 '''
                    if content_list[a] == '项目' or \
                            (content_list[a][-2:] == '年度' and content_list[a][4:6] == '年度'):
                        a += 1
                        continue
                    ''' 去除表格下方，‘注’的内容 '''
                    if content_list[a][:2] == "注：":
                        # 因表格提取时不会有句号，所以需要保证在‘注’以上的正文需要以句号为结尾
                        if len(text) > 0:
                            while len(text) > 0 and text[-1] != '。':
                                text = text[:len(text)-1]
                        a = in_note(content_list, a)
                        if a == len(content_list):
                            break
                    ''' 截取两个标题之间的正文内容 '''
                    if istitle(content_list, a):                    # 第a行是标题
                        if not_over:                                # 上一页的记录正文未结束
                            ''' 去除正文的最后含有表格内未匹配信息没有句号的内容 '''
                            if '。' in text:
                                if len(text) > 0:
                                    while len(text) > 0 and text[-1] != '。':
                                        text = text[:len(text) - 1]
                                    if '。' not in text[:-1] and len(text) > 600:
                                        text = ''
                                ft.write(text.replace(' ', ''))     # 储存正文
                                ft.write('\n')
                            text = ''
                            not_over = False                        # 标志记录正文完毕
                            continue                                # 下面不用执行，a不用加1

                        text = ''
                        b = a + 1
                        while b < len(content_list):
                            ''' 属于表格内的信息，则跳过（格式简单的表格，一个格内只有一行文字） '''
                            if in_table(content_list, b, table_list):
                                b += 1
                                continue
                            ''' 表格中遗漏掉的、无法匹配的内容 '''
                            if content_list[b] == '项目' or \
                                    (content_list[b][-2:] == '年度' and content_list[b][4:6] == '年度'):
                                b += 1
                                continue
                            ''' 去除表格下方，‘注’的内容 '''
                            if content_list[b][:2] == "注：":
                                if len(text) > 0:
                                    while len(text) > 0 and text[-1] != '。':
                                        text = text[:len(text) - 1]
                                    if '。' not in text[:-1] and len(text) > 600:
                                        text = ''
                                b = in_note(content_list, b)
                                # 如果注中的内容已到达本页最后
                                if b == len(content_list):
                                    break
                            if istitle(content_list, b):                # b行是标题
                                ''' 去除正文的最后含有表格内未匹配信息的内容 '''
                                if '。' in text:
                                    if len(text) > 0:
                                        while len(text) > 0 and text[-1] != '。':
                                            text = text[:len(text) - 1]
                                        if '。' not in text[:-1] and len(text) > 600:
                                            text = ''
                                    ft.write(text.replace(' ', ''))     # 所记录的正文储存起来
                                    ft.write('\n')
                                text = ''                               # text置空
                                a = b - 1                               # 调整a
                                break
                            else:                                       # b行不是标题
                                if '√' in content_list[b] or '□' in content_list[b]:
                                    b += 1
                                    continue
                                text += content_list[b]
                            b += 1

                        if b == len(content_list):                  # 此页最后一行没有标题，只有正文
                            not_over = True
                            break
                    else:                                   # a处不是标题
                        # 跳过对号和方块
                        if '√' in content_list[a] or '□' in content_list[a]:
                            a += 1
                            continue
                        if not_over:                        # 上文记录未结束
                            text += content_list[a]
                    a += 1
                if j == end_page - 2:
                    ''' 去除正文的最后含有表格内未匹配信息的内容 '''
                    if '。' in text:
                        if len(text) > 0:
                            while len(text) > 0 and text[-1] != '。':
                                text = text[:len(text) - 1]
                            if '。' not in text[:-1] and len(text) > 600:
                                text = ''
                        ft.write(text.replace(' ', ''))
            except RuntimeError as r:
                # 出错的页面跳过
                print(r)
    #             text = ''
    #             continue
    #     ft.write('\n')
    # ft.close()


# 第一个入口，获取目录，然后查找”讨论与分析“，”重要事项“，”变动和股份情况“的页数范围，然后转到 triple_extra
def pdf2txt(pdf_file_path, txt_file_path):
    """
    --------------读取 pdf 并写入 txt---------------
    """
    global parser_pdf_count, cannot_parser_count


    '''-------------- 各部分开始和结束的页数 --------------'''
    begin_page1, end_page1 = 0, 0   # 讨论与分析
    begin_page2, end_page2 = 0, 0   # 重要事项
    begin_page3, end_page3 = 0, 0   # 股份变动与股东情况
    end1 = False                    # 第一个模块抽取完毕标识
    end2 = False                    # 第二个模块抽取完毕标识
    end3 = False                    # 第三个模块抽取完毕标识
    '''-------------- 分析pdf，同时捕获异常 --------------'''
    try:
        pdf = pdfplumber.open(pdf_file_path)        # 使用库函数打开pdf文件
    except:
        return -1
    else:
        ''' --------------- 正文文件名 --------------- '''
        out_txt_name = txt_file_path

        # 1.页面对齐，在前 10页中找到 pdf标注的第 1页
        add_page = 0
        try:
            for k in range(10):
                content = pdf.pages[k].extract_text().replace(' ', '').replace(' ', '')
                c_list2 = [x for x in content.split('\n') if x != '']  # 去除列表里的空值
                if c_list2:
                    if c_list2[-1] == '1':
                        add_page = k
                        break
        except Exception as e:
            return -3

        '''-------------自动获取页面-------------'''
        # 2.获取pdf文件全部页数
        page_num = len(pdf.pages)

        # 3.逐页解析（暂时设定在前20页里寻找目录页）
        catalog = ''
        catalog_flag = False
        '''--------------获取目录页内容begin--------------'''
        try:
            for i in range(20):
                # 获取pdf第i页的内容
                string = pdf.pages[i].extract_text().replace(' ', '').replace(' ', '')
                c_list = [x for x in string.split('\n') if x != '']  # 获得目录列表，并去除列表里的空值
                # 4.1定位目录页
                if '目录' in c_list or '目目目目录录录录' in c_list or in_punc(string):
                    catalog = string
                    # 有些老旧的 pdf页面文字解析为 cid
                    if '(cid:' in catalog:
                        return -5
                    '''-------------------目录跨越多个页面-------------------'''
                    # 年报目录最多跨两页，未容错，扩充到查找三页
                    for j in range(i + 1, i + 3):
                        catalog_extend = pdf.pages[j].extract_text().replace(' ', '').replace(' ', '') # pdf第j页的内容
                        # 判断是否存在目录符号
                        if in_punc(catalog_extend):
                            catalog += catalog_extend
                            j += 1
                        else:
                            catalog_flag = True
                            break
                if catalog_flag:
                    break
            if catalog == '':
                print("未找到目录页")
                return -4
        except Exception as e:
            return -3
        '''--------------获取目录页内容 end --------------'''

        c_list = [x for x in catalog.split('\n') if x != '']
        '''--------------解析目录内容，定位获取各部分的页号范围--------------'''
        c = 0
        while c < len(c_list):
            '''--------------  讨论与分析  --------------'''
            if not end1 and ("讨论与分析" in c_list[c]) or "董事会报告" in c_list[c]:      # 目录中存在讨论与分析或董事会报告
                # 获取的文字目录中标题与 ’........‘ 是分行的
                tag1 = "讨论与分析"
                if "董事会报告" in c_list[c]:
                    tag1 = "董事会讨论与分析报告"
                if not in_punc(c_list[c]) and in_punc(c_list[c+1]):
                    c += 1
                # 起始页号
                begin_page1 = get_page_num(c_list[c])
                # 应对特殊情况，部分老版pdf的抽取，在目录页面，页号在'-----'符号下面一行显示
                if begin_page1 == 0:
                    begin_page1 = get_page_num(c_list[c-1])
                # 若已经看到最后一个目录，那么终止页号也就是最后一页
                if c == len(c_list) - 1:
                    end_page1 = page_num  # 终止页号
                else:
                    if not in_punc(c_list[c-1]) and in_punc(c_list[c]):
                        c += 1
                    end_page1 = get_page_num(c_list[c + 1])  # 终止页号

                if begin_page1 == 0 and end_page1 == 0:
                    c += 1
                    continue
                if begin_page1 > end_page1 or begin_page1 > page_num or end_page1 > page_num \
                        or (end_page1 - begin_page1) > 200:
                    print("页号范围异常")
                    return -2
                c += 1
                '''-------------解析具体页面-------------'''
                text_extra(tag1,
                             add_page,
                             begin_page1,
                             end_page1,
                             pdf,
                             out_txt_name)
                if begin_page1 != 0 and end_page1 != 0:
                    end1 = True
            '''--------------  重大事项  --------------'''
            if not end2 and ("重大" in c_list[c] or "重要" in c_list[c]) and "事项" in c_list[c]:
                # 获取的文字目录中标题与’........‘是分行的
                tag2 = "重大事项"
                if "重要" in c_list[c]:
                    tag2 = "重要事项"
                if not in_punc(c_list[c]) and in_punc(c_list[c+1]):
                    c += 1
                # 起始页号
                begin_page2 = get_page_num(c_list[c])
                if begin_page2 == 0:
                    begin_page2 = get_page_num(c_list[c-1])
                # 若已经看到最后一个目录，那么终止页号也就是最后一页
                if c == len(c_list) - 1:
                    end_page2 = page_num  # 终止页号
                else:
                    if not in_punc(c_list[c-1]) and in_punc(c_list[c]):
                        c += 1
                    end_page2 = get_page_num(c_list[c + 1])  # 终止页号
                if begin_page2 == 0 and end_page2 == 0:
                    c += 1
                    continue
                # print('重要事项', begin_page2, end_page2-1)
                if begin_page2 > end_page2 or begin_page2 > page_num or end_page2 > page_num \
                        or (end_page2 - begin_page2) > 200:
                    return -2
                '''-------------对具体页面范围进行三元组抽取-------------'''
                text_extra(tag2,
                             add_page,
                             begin_page2,
                             end_page2,
                             pdf,
                             out_txt_name)
                if begin_page2 != 0 and end_page2 != 0:
                    end2 = True
            '''--------------  股东情况变动  --------------'''
            if not end3 and ("变动" in c_list[c] or "股东" in c_list[c]) and "情况" in c_list[c]:
                # 获取的文字目录中标题与’........‘是分行的
                if not in_punc(c_list[c]) and in_punc(c_list[c+1]):
                    c += 1
                # 起始页号
                begin_page3 = get_page_num(c_list[c])
                if begin_page3 == 0:
                    begin_page3 = get_page_num(c_list[c-1])
                # 若已经看到最后一个目录，那么终止页号也就是最后一页
                if c == len(c_list) - 1:
                    end_page3 = page_num
                # 标题包含“情况”的章节进行抽取
                else:
                    if in_punc(c_list[c]):
                        if in_punc(c_list[c+1]):
                            while '情况' in c_list[c]:
                                c += 1
                        else:
                            while '情况' in c_list[c-1]:
                                c += 2
                    else:
                        while '情况' in c_list[c]:
                            c += 1
                    end_page3 = get_page_num(c_list[c])  # 终止页号
                if end_page3 == 0:
                    end_page3 = get_page_num(c_list[c-1])  # 终止页号
                if begin_page3 == 0 and end_page3 == 0:
                    c += 1
                    continue
                # print('股本变动与股东情况', begin_page3, end_page3-1)
                if begin_page3 > end_page3 or begin_page3 > page_num or end_page3 > page_num \
                        or (end_page3 - begin_page3) > 200:
                    return -2
                '''-------------对具体页面范围进行三元组抽取-------------'''
                text_extra('股本变动与股东情况',
                             add_page,
                             begin_page3,
                             end_page3,
                             pdf,
                             out_txt_name)
                if begin_page3 != 0 and end_page3 != 0:
                    end3 = True
            c += 1
            if (begin_page1 != 0 and end_page1 != 0) or \
                    (begin_page2 != 0 and end_page2 != 0) or \
                    (begin_page3 != 0 and end_page3 != 0):
                break
        if begin_page1 == 0 and begin_page2 == 0 and begin_page3 == 0 and end_page1 == 0 and end_page2 == 0 and end_page3 == 0:
            return -4
    return 1


def convert_pdf_to_txt(pdf_file_path, txt_file_path):
    print(txt_file_path)

    flag = pdf2txt(pdf_file_path, txt_file_path)
    if flag == 0:
        return False, "不是pdf"
    elif flag == -1:
        return False, "疑似pdf文件格式问题"
    elif flag == -2:
        return False, "目录页号问题"
    elif flag == -3:
        return False, "抽取异常"
    elif flag == -4:
        return False, "未找到目录或目录格式不对"
    elif flag == -5:
        return False, "抽取格式错误(cid:xx)"
    else:
        return True, "抽取完毕！"

