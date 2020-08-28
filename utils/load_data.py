# -*-coding:utf8-*-
import sys
import os
import math
import collections

# import tools

from numpy import array



def _sent2array_ent(sent, wdict, edict, ydict, max_len):
    words = map(lambda x: wdict.get(x, wdict['OTHER-WORDS-ID']), sent[0])
    ents = map(lambda x: edict.get(x, edict['NEGATIVE']), sent[1])
    # print(words)
    # print(ents)
    # print(list(words))
    # print(sent[0])
    # print(sent[1])
    # print(sent[2])
    unknownWordindex=wdict['OTHER-WORDS-ID']

    MAX_SEN_LEN = max_len
    words=list(words)
    ents=list(ents)
    if len(words) < MAX_SEN_LEN:
        words += ([unknownWordindex] * (MAX_SEN_LEN - len(words)))
        ents += ([edict['NEGATIVE']] * (MAX_SEN_LEN - len(ents)))
    elif len(words) > MAX_SEN_LEN:
        words = words[:MAX_SEN_LEN]
        ents = ents[:MAX_SEN_LEN]
    print('cur sentence',sent[2])
    labels= [ydict.get(x, 'NEGATIVE') for x in sent[2]]
    print('labels ',labels)
    
    return words, ents, labels

def load_data_ent(data_path, wdict, edict, ydict, max_len):
    sen, ent, y = [], [], []
    
    for line in open(data_path):
        line = line.strip()
        if not line: continue
        if len(line.split('\t')) < 3: continue
        wds = line.split('\t')[:-1]
        wds, ents = zip(*[x.split(' ') for x in wds])
        ls = line.split('\t')[-1].strip().lower().split(' ')
        words, ents, labels = _sent2array_ent((wds, ents, ls), wdict, edict, ydict, max_len)
        sen.append(words)
        ent.append(ents)
        y.append(labels)

    # return [array(sen, dtype='int32'), array(ent, dtype='int32'), y]
    return [array(sen, dtype='int32'), array(ent, dtype='int32'), y]
#


        


