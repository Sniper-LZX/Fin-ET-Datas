#!/usr/bin/env python3
# coding: utf-8
# Time    : 2024/7/6 14:54
# Author  : SJ_Sniper
# File    : Sentence_Parser_Inter.py
# Description :
import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller


class LtpParser:
    ''' 加载模型 '''
    def __init__(self, LTP_DIR):
        # 句子分词
        self.segmentor = Segmentor(os.path.join(LTP_DIR, "cws.model"))
        # 词性标注
        self.postagger = Postagger(os.path.join(LTP_DIR, "pos.model"))

    def parser_main(self, sentence):
        # 分词
        words = list(self.segmentor.segment(sentence))
        # 词性标注   ['nh', 'n', 'nt', 'v', 'n', 'u', 'wp', 'r', 'v', 'd', 'a']
        postags = list(self.postagger.postag(words))
        return postags
