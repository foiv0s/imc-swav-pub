import torch
import numpy as np
from utils import shoot_infs_v2

eps = 1e-17


def entropy(x1, x2=None):
    x2 = x1 if x2 is None else x2
    return -(x1 * torch.log(x2.clamp(eps, 1.)))


def mi(y, u, b=4, a=1e-2):
    lgt_reg = a * torch.relu(torch.abs(y) - 5.).sum(-1).mean()
    py, pu = torch.softmax(y, -1), torch.softmax(u, -1)
    p_yu = torch.matmul(py.T, pu)  # k x k’
    p_yu /= p_yu.sum()  # normalize to sum 1
    p_u = p_yu.sum(0).view(1, -1)  # marginal p_u
    p_y = p_yu.sum(1).view(-1, 1)  # marginal p_y
    h_uy = (p_yu * (torch.log(p_u) - torch.log(p_yu))).sum()  # conditional entropy
    hy = b * (p_yu * torch.log(p_y)).sum()  # weighted marginal entropy
    return h_uy + hy, lgt_reg


def sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs_v2(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda() / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda() / (-1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs_v2(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q)).float()
