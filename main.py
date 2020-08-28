#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/31 15:29
# @Author : Aries
# @Site :
# @File : main2.py
# @Software: PyCharm
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
import random

batch_size = 500
eval_batch_size = batch_size
unlabeled_batch_size = batch_size
num_labeled = 1000
num_valid = 1000
num_iter_per_epoch = int(50000/10 / (batch_size))
print('per epoch num iter', num_iter_per_epoch)
eval_freq = 2
lr = 0.001
cuda_device = "1,2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--use_vat', type=bool, default=True)
parser.add_argument('--method', default='vat')
parser.add_argument('--output', default='./output_vat.pkl')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def train(model, x, y, ul_x, optimizer, use_vat):
    ce = nn.CrossEntropyLoss()
    y_pred = model(x)
    ce_loss = ce(y_pred, y)
    v_loss = 0.0
    # if use_vat==True:
    ul_y = model(ul_x)
    v_loss = vat_loss(model, ul_x, ul_y, eps=opt.epsilon)
    loss = v_loss + ce_loss
    # if opt.method == 'vatent':
    #     loss += entropy_loss(ul_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return v_loss, ce_loss


def eval(model, x, y):
    y_pred = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

elif opt.dataset == 'cifar10':
    num_labeled = 4000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])),
        batch_size=eval_batch_size, shuffle=True)



else:
    raise NotImplementedError

train_data = []
train_target = []
import pdb

valid_data = []
valid_target = []

test_data = []
test_target = []
# 生成随机种子
# np_index_data = np.random.choice(int(50000/batch_size), int(50000/batch_size/10), replace=False)
# np.save(np_index_data,'./random_shuffle.seed')
np_index_data=np.load('./random_shuffle.seed.npy')
index =0
for (data, target) in train_loader:
    # 随机取一半数据
    if index in np_index_data:
        train_data.append(data)
        train_target.append(target)
    else:
        valid_data.append(data)
    index += 1
    # print(index)

for (data, target) in test_loader:
    test_data.append(data)
    test_target.append(target)

# pdb.set_trace()

# random.shuffle(train_data)
# random.shuffle(train_target)
# def generate_random_index(max_index,min_index,num):
#     random_list=[]
#     while 1:
#         if len(random_list)==num:
#             return  random_list
#         r=random.randint(0,100)
#         if r not in random_list:
#             random_list.append(r)
#
# random_list = generate_random_index(50000,0,20000)
# 一共5万张  50000/batch_size   = len(train_data)


# 取2.5万张做l数据
# 取2.5万张做ul数据
# 1万张测试数据

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)
valid_data = torch.cat(valid_data, dim=0)
# train_random_data= [train_data[i,] for i in random_list]
#
# train_random_target = [train_target[i,] for i in random_list]
#
# valid_data=[train_data[i] for i in range(50000) if i not in random_list]


# 按照比例划分
# num_valid=int(len(train_data)/10*6)
# num_labeled=int(len(train_data)/10*6)
# valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
# valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]
#
print('len label train data ', len(train_data))
print('len unlabel train data ', len(valid_data))

labeled_train, labeled_target = train_data, train_target
unlabeled_train = valid_data

model = tocuda(VAT(opt.top_bn))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)

min_loss = 10.0

# train the network
for epoch in range(opt.num_epochs):

    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

    for i in range(num_iter_per_epoch):
        # pdb.set_trace()
        # batch_indices = torch.LongTensor(np.random.choice(len(labeled_train), 9, replace=False))
        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(
            np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

        v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                optimizer, use_vat=opt.use_vat)

        if i % 10 == 0:
            # import pdb
            # pdb.set_trace()
            print("Epoch :", epoch, "Iter :", i, "VAT Loss :", str(v_loss), "CE Loss :", ce_loss.data.item())

    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
        print("Train accuracy :", train_accuracy.data[0])

        test_accuracy = 0.0
        counter = 0
        for (data, target) in test_loader:
            n = data.size()[0]
            acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
            test_accuracy += n * acc
            counter += n
        print("Full test accuracy :", test_accuracy.data[0] / counter)

test_accuracy = 0.0
counter = 0
for (data, target) in test_loader:
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n * acc
    counter += n

print('use vat result:')
print("Full test accuracy :", test_accuracy.data[0] / counter)
torch.save(model.state_dict(), opt.output)
