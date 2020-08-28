#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/4 15:01
# @Author : Aries
# @Site : 
# @File : generate_vat_data.py
# @Software: PyCharm
import random

data_line = []

with open('./train.txt', 'r', encoding='utf-8') as fw:
    for line in fw:
        data_line.append(line.strip())
random.shuffle(data_line)

with open('./train_split_1.txt', 'w', encoding='utf-8') as fw:
    for line in data_line[0:int(len(data_line) / 2)]:
        fw.write(line)
        fw.write('\n')

with open('./train_split_2.txt', 'w', encoding='utf-8') as fw:
    for line in data_line[int(len(data_line) / 2):]:
        fw.write(line)
        fw.write('\n')
