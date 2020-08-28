#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/13 19:46
# @Author : Aries
# @Site : 
# @File : model_ed.py
# @Software: PyCharm
import torch.nn as nn
import torch

class EventDetation(nn.Module):

      def __init__(self,opt, embedings=None):

            super(EventDetation, self).__init__()
            hidden_dim = opt.emb_dim + opt.dim_ent
            self.opt=opt
            lengths = hidden_dim

            # word embedding
            if opt.embed_type=='baseline':
                # 随机初始化
                self.w_embedding = nn.Embedding(opt.word_count, opt.emb_dim)
            else:
                self.w_embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedings),freeze=True)
            # self.w_embedding=nn.Embedding(20196,768)
            self.w_embedding.weight.requires_grad = True

            # entity embedding
            self.Ent_embedding=nn.Embedding(opt.n_ent, opt.dim_ent)
            self.Ent_embedding.weight.requires_grad = True

            self.evt_embedding  =nn.Embedding(opt.n_class, hidden_dim)
            self.evt_embedding.weight.requires_grad = True

            self.evt_embedding_last= self.evt_embedding
            self.evt_embedding_last.weight.requires_grad = True

            # x_e = Ent_embedding(ent)
            # x_evt =evt_embedding(evt)
            # x_w=w_embedding(sent)
            # x_evt_last=evt_embedding_last(evt)
            # x = torch.cat([x_w, x_e], 2)
            # gru
            self.lstm=nn.LSTM(input_size=lengths, hidden_size=int(hidden_dim), num_layers=2)

            # self.h0 = torch.randn(2, settings['batch_size'], lengths)
            # self.c0 = torch.randn(2,  settings['batch_size'], lengths)

      def find_wd(self,sent):
            x_w = self.w_embedding(sent)
            return x_w

      def forward(self, batch_sent_wd,
            batch_evt,
            batch_mask,
            batch_en):
            sent, evt, mask, ent  = batch_sent_wd,batch_evt,batch_mask,batch_en
            hidden_dim = self.opt.emb_dim + self.opt.dim_ent
            x_w = sent
            ent = ent
            mask = mask
            evt = evt
            x_e = self.Ent_embedding(ent)
            x_evt =self.evt_embedding(evt)
            x_evt_last=self.evt_embedding_last(evt)
            # print(x_evt_last.size())
            x = torch.cat([x_w, x_e], 2)
            x=x.permute(1,0,2)
            import pdb
            # pdb.set_trace()
            # gru
            # print(x.size())
            output, (cell, hidden)= self.lstm(x)
            # print(output.size(), cell.size(),hidden.size())
            # attention
            # attention1_logits = torch.matmul(output, torch.transpose(x_evt, [0, 2, 1]))
            attention1_logits=torch.matmul(output.permute(1,0,2), x_evt.permute(0,2,1))
            #三维向量转2维向量
            attention1_logits = torch.reshape(attention1_logits, [-1, self.opt.max_l]) * mask.float()
            attention1 = torch.exp(attention1_logits) * mask.float()


            # attention score
            _score1 = attention1_logits * attention1 / attention1.sum(dim=1,keepdim=True)
            score1 = _score1.sum(dim=1,keepdim=True)

            # global score
            x_evt_last = torch.reshape(x_evt_last, [-1, hidden_dim])

            score2=(hidden[1] * x_evt_last).sum(dim=1,keepdim=True)

            alpha = self.opt.alpha
            score = score1 * alpha + score2 * (1 - alpha)
            # print('score',score)

            return score