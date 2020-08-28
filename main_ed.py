#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/13 20:31
# @Author : Aries
# @Site : 
# @File : main_ed.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/31 15:29
# @Author : Aries
# @Site :
# @File : main2.py
# @Software: PyCharm
import argparse
import torch.optim as optim
from model_ed import *
from utils import ed_utils, load_data, show_result, tools
from torch.autograd import Variable

import argparse
import torch.optim as optim
from model import *
from utils import *
import os
from utils import *
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import json
from utils import tools, show_result

Prifix = os.path.join(os.getcwd(), os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--max_l', type=int, default=40)
parser.add_argument('--n_class', type=int, default=34)
parser.add_argument('--n_ent', type=int, default=55)
parser.add_argument('--dim_ent', type=int, default=50)
parser.add_argument('--n_eps', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_type', default='glove')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--l2_weight', type=float, default=0.00001)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--use_vat', type=bool, default=False)
parser.add_argument('--method', default='vat')
parser.add_argument('--output', default='./output/glove_baseline.bin')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def train(model, batch_sent_wd,
          batch_evt,
          batch_mask,
          batch_en,
          y,
          ul_batch_sent,
          ul_extra_evt,
          ul_batch_mask,
          ul_batch_ent,
          optimizer, use_vat):
    y_pred = model(batch_sent_wd, batch_evt, batch_mask, batch_en)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    mse_loss = loss_fn(y_pred.float(), y.float())
    # l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)
    # for param in model.parameters():
    #     l1_regularization += torch.norm(param, 1)
    #     l2_regularization += torch.norm(param, 2)
    v_loss = 0
    if use_vat == True:
        ul_y = model(ul_batch_sent, ul_extra_evt, ul_batch_mask, ul_batch_ent)
        v_loss = ed_utils.vat_loss(model, ul_batch_sent, ul_extra_evt, ul_batch_mask, ul_batch_ent, ul_y,
                                   eps=opt.epsilon)
    loss = mse_loss + v_loss
    # if opt.method == 'vatent':
    #     loss += entropy_loss(ul_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return v_loss, mse_loss


def load_dicts(path):
    ans = json.loads(open(path).read())
    return ans


def predict_sen(model, sen, ent, ydict, max_ans=3):
    ans = []
    sens, ents, evts, masks = [], [], [], []
    labels = list(ydict.values())
    for y in labels:
        sens.append(sen)
        ents.append(ent)
        masks.append([1 if x >= 0 else 0 for x in sen])
        evts.append(y)
    sens = np.array(sens)
    evts = np.array(evts)
    masks = np.array(masks)
    evts = evts[:, np.newaxis]
    sens = model.find_wd(torch.LongTensor(sens))
    pred = model(Variable(sens), Variable(torch.LongTensor(evts)), Variable(torch.LongTensor(masks)),
                 Variable(torch.LongTensor(ents)))
    for y, p in zip(labels, pred):
        if y != ydict['negative'] and p > 0.5:
            ans.append((y, p))
        if y == ydict['negative'] and p < 0.5:
            # print s
            pass
    ans = sorted(ans, key=lambda a: a[1], reverse=True)
    # print('ans is ', ans)
    ret = []
    if len(ans) > 0:
        for k in ans[:max_ans]:
            ret.append(k[0])
    else:
        ret.append(ydict['negative'])
    return ret


def convert2binary(data, ydict, neg_prob=0.4):
    sen, ent, y = data
    ret_sen, ret_ent, ret_evt, ret_label, ret_mask = [], [], [], [], []
    for idx in range(len(sen)):
        for ly in ydict.values():
            lb = 1 if ly in y[idx] else 0
            if lb == 0 and random.random() > neg_prob: continue
            ret_sen.append(sen[idx])
            ret_ent.append(ent[idx])
            ret_evt.append(ly)
            ret_label.append(lb)
            ret_mask.append([1 if x >= 0 else 0 for x in sen[idx]])
    return np.array(ret_sen), np.array(ret_ent), np.array(ret_evt), np.array(ret_label), np.array(ret_mask,
                                                                                                  dtype='float32')


train_path = '%s/data/train_split_1.txt' % Prifix
ul_train_path = '%s/data/train_split_2.txt' % Prifix
test_path = '%s/data/test.txt' % Prifix
edict_path = '%s/data/ent_dict.txt' % Prifix

ydict_path = '%s/data/label_dict.txt' % Prifix
wdict_path=''
WordsEmbedings = None
word_dest_p = ''
if opt.embed_type == 'bert':
    word_dest_p = '%s/data/embeddings/word_vec_bert.txt' % Prifix
    WordsEmbedings = tools.load_embedding(word_dest_p)
    wdict_path = '%s/data/word_dict_bert.txt' % Prifix
elif opt.embed_type == 'glove':
    word_dest_p = '%s/data/embeddings/word_vec_glove.txt' % Prifix
    WordsEmbedings = tools.load_embedding(word_dest_p)
    wdict_path='%s/data/word_dict_glove.txt' % Prifix
else:
    wdict_path = '%s/data/word_dict_bert.txt' % Prifix

wdict = tools.load_dict(wdict_path)
edict = tools.load_dict(edict_path)
ydict = tools.load_dict(ydict_path)
ydict = {k.lower(): v for k, v in ydict.items()}

opt.word_count = len(wdict.keys())
l_train_data = load_data.load_data_ent(train_path, wdict, edict, ydict, opt.max_l)

ul_train_data = load_data.load_data_ent(ul_train_path, wdict, edict, ydict, opt.max_l)

test_data = load_data.load_data_ent(test_path, wdict, edict, ydict, opt.max_l)

t_train_sen, t_train_ent, t_train_evt, t_train_y, t_train_mask = convert2binary(l_train_data, ydict)
ul_t_train_sen, ul_t_train_ent, ul_t_train_evt, ul_t_train_y, ul_t_train_mask = convert2binary(ul_train_data, ydict)

test_sen, test_ent, test_evt, test_y, test_mask = convert2binary(test_data, ydict)

print(WordsEmbedings)
print(WordsEmbedings.shape)
model = tocuda(EventDetation(opt, WordsEmbedings))
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0001)

batch_size = opt.batch_size

#########生成训练数据############
if len(t_train_sen) % batch_size > 0:
    extra_size = batch_size - len(t_train_sen) % batch_size
    rand_train = np.random.permutation(range(len(t_train_sen)))[:extra_size]
    extra_y = t_train_y[rand_train]
    extra_sen = t_train_sen[rand_train]
    extra_evt = t_train_evt[rand_train]
    extra_ent = t_train_ent[rand_train]
    extra_mask = t_train_mask[rand_train]
    t_train_y = np.concatenate((t_train_y, extra_y))
    t_train_evt = np.concatenate((t_train_evt, extra_evt))
    t_train_ent = np.concatenate((t_train_ent, extra_ent))
    t_train_sen = np.concatenate((t_train_sen, extra_sen))
    t_train_mask = np.concatenate((t_train_mask, extra_mask))
if len(ul_t_train_sen) % batch_size > 0:
    extra_size = batch_size - len(ul_t_train_sen) % batch_size
    rand_train = np.random.permutation(range(len(ul_t_train_sen)))[:extra_size]
    ul_extra_y = ul_t_train_y[rand_train]
    ul_extra_sen = ul_t_train_sen[rand_train]
    ul_extra_evt = ul_t_train_evt[rand_train]
    ul_extra_ent = ul_t_train_ent[rand_train]
    ul_extra_mask = ul_t_train_mask[rand_train]
    ul_t_train_y = np.concatenate((ul_t_train_y, ul_extra_y))
    ul_t_train_evt = np.concatenate((ul_t_train_evt, ul_extra_evt))
    ul_t_train_ent = np.concatenate((ul_t_train_ent, ul_extra_ent))
    ul_t_train_sen = np.concatenate((ul_t_train_sen, ul_extra_sen))
    ul_t_train_mask = np.concatenate((ul_t_train_mask, ul_extra_mask))
if len(test_sen) % batch_size > 0:
    extra_size = batch_size - len(test_sen) % batch_size
    rand_train = np.random.permutation(range(len(test_sen)))[:extra_size]
    test_extra_y = test_y[rand_train]
    test_extra_sen = test_sen[rand_train]
    test_extra_evt = test_evt[rand_train]
    test_extra_ent = test_ent[rand_train]
    test_extra_mask = test_mask[rand_train]
    test_y = np.concatenate((test_y, test_extra_y))
    test_evt = np.concatenate((test_evt, test_extra_evt))
    test_ent = np.concatenate((test_ent, test_extra_ent))
    test_sen = np.concatenate((test_sen, test_extra_sen))
    test_mask = np.concatenate((test_mask, test_extra_mask))
n_batchs = int(len(t_train_y) / batch_size)
print('n_batchs', n_batchs)
test_batchs = int(len(test_sen) / batch_size)
# train the network
for epoch in range(opt.num_epochs):
    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * opt.lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        # 生成数据
    for k in range(n_batchs):
        # label data
        shuff_l = torch.LongTensor(np.random.choice(len(t_train_sen), batch_size, replace=False))
        # print(shuff_l)
        shuff_ul = torch.LongTensor(np.random.choice(len(ul_t_train_sen), batch_size, replace=False))
        batch_sent = t_train_sen[shuff_l]
        batch_evt = t_train_evt[shuff_l]
        batch_ent = t_train_ent[shuff_l]
        batch_y = t_train_y[shuff_l]
        batch_evt = batch_evt[:, np.newaxis]
        batch_y = batch_y[:, np.newaxis]
        batch_mask = t_train_mask[shuff_l]
        #
        batch_sent_wd = model.find_wd(torch.LongTensor(batch_sent))

        # unlabel data
        ul_batch_sent = ul_t_train_sen[shuff_ul]
        ul_batch_evt = ul_t_train_evt[shuff_ul]
        ul_batch_ent = ul_t_train_ent[shuff_ul]
        ul_batch_evt = ul_batch_evt[:, np.newaxis]
        ul_batch_mask = ul_t_train_mask[shuff_ul]
        ul_batch_sent_wd = model.find_wd(torch.LongTensor(ul_batch_sent))
        #   训练
        batch_sent_wd = torch.FloatTensor(batch_sent_wd)
        batch_evt = torch.LongTensor(batch_evt)
        batch_mask = torch.LongTensor(batch_mask)
        batch_ent = torch.LongTensor(batch_ent)

        y = torch.LongTensor(batch_y)

        # 未标记数据
        ul_batch_sent = torch.LongTensor(ul_batch_sent)
        ul_batch_evt = torch.LongTensor(ul_batch_evt)
        ul_batch_mask = torch.LongTensor(ul_batch_mask)
        ul_batch_ent = torch.LongTensor(ul_batch_ent)
        v_loss, mse_loss = train(model.train(), Variable(tocuda(batch_sent_wd)), Variable(tocuda(batch_evt)),
                                 Variable(tocuda(batch_mask)), Variable(tocuda(batch_ent)),
                                 Variable(tocuda(y)),
                                 Variable(tocuda(ul_batch_sent_wd)), Variable(tocuda(ul_batch_evt)),
                                 Variable(tocuda(ul_batch_mask)), Variable(tocuda(ul_batch_ent)),
                                 optimizer, use_vat=opt.use_vat)
        if k % 10 == 0:
            # import pdb
            # pdb.set_trace()
            print("Epoch :", epoch, "Iter :", k, "VAT Loss :", v_loss, "mse_loss :", mse_loss.data.item())
test_accuracy = 0.0
test_sents, test_ents, test_y = test_data
n_test_batch = len(test_sents)
t_result = []
for k in range(n_test_batch):
    #if k > 10: break
    pred = predict_sen(model.eval(), test_sents[k], test_ents[k], ydict)
    t_result.append((pred, test_y[k]))
ptr_str, f = show_result.evaluate_results_binary(t_result, ydict['negative'])
print(ptr_str)
torch.save(model.state_dict(), opt.output)
