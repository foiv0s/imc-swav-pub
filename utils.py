import torch
import numpy as np
from stats import update_train_accuracies


# Evaluate accuracy on test set
def test_model(model, test_loader, device, stats):
    # warm up norm layers
    _warmup_batchnorm(model, test_loader, device, batches=50, train_loader=False)

    model.eval()
    targets, predictions = [], []
    for _, (images, targets_, idxs) in enumerate(test_loader):
        images = images.to(device)
        val_idxs = targets_ >= 0
        with torch.no_grad():
            res_dict = model(x=images, eval_only=True)
        predictions.append(res_dict['y'][val_idxs].cpu().numpy()), targets.append(targets_[val_idxs])
    targets, predictions = np.concatenate(targets).ravel(), np.concatenate(predictions).ravel()
    model.train()
    update_train_accuracies(stats, targets, predictions, 'Test Clustering ')


def _warmup_batchnorm(model, data_loader, device, batches=50, train_loader=False):
    model.train()
    for i, (images, _, idxs) in enumerate(data_loader):
        if i == batches:
            break
        if train_loader:
            images = images[0]
        _ = model(x=images.to(device), eval_only=True)


'''
Following two methods (distributed_sinkhorn, shoot_infs) are based on SwAV implementation
credits to https://github.com/facebookresearch/swav
'''


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
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs_v2(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    if mask_inf.sum() > 0.:
        inp_tensor[mask_inf] = 0
        m = torch.max(inp_tensor)
        inp_tensor[mask_inf] = m
    return inp_tensor
