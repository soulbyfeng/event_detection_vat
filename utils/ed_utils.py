import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(model, ul_batch_sent,ul_extra_evt ,ul_batch_mask ,ul_batch_ent , ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv
    # 获取内部词向量作为实际的ul_x
    sent=ul_batch_sent
    d = torch.Tensor(ul_batch_sent.size()).normal_()

    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        # d = Variable(d.cuda(), requires_grad=True)
        d=Variable(d,requires_grad=True)
        ul_batch_sent=ul_batch_sent+d
        y_hat = model( ul_batch_sent,ul_extra_evt ,ul_batch_mask ,ul_batch_ent)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()
    d = _l2_normalize(d)
    # d = Variable(d.cuda())
    d = Variable(d)
    r_adv = eps *d
    # compute lds
    sent = sent + r_adv.detach()
    y_hat = model(sent,ul_extra_evt ,ul_batch_mask ,ul_batch_ent)
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)