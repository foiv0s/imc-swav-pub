import sys
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from utils import test_model
from stats import AverageMeterSet, update_train_accuracies


def _train(model, optimizer, scheduler_inf, train_loader, test_loader, nmb_crops, stat_tracker,
           checkpointer, device, warmup, epochs):
    '''
    Training loop for optimizing overall framework
    '''
    lr_real = optimizer.param_groups[0]['lr']
    torch.cuda.empty_cache()

    # If mixed precision is on, will add the necessary hooks into the model
    # and optimizer for half() conversions
    next_epoch = checkpointer.get_current_position()
    total_updates = next_epoch * len(train_loader)
    # run main training loop
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        time_epoch = time.time()
        targets, predictions = [], []
        model.reset_membank_list()
        for _, ((aug_imgs, raw_imgs), targets_, idx) in enumerate(train_loader):

            # Perform clustering only on label idxs
            val_idxs = targets_ >= 0
            targets.append(targets_[val_idxs].numpy())
            aug_imgs = [aug_img.to(device) for aug_img in aug_imgs]

            res_dict = model(x=aug_imgs, eval_only=False, nmb_crops=nmb_crops, eval_idxs=val_idxs)

            # Warmup
            if total_updates < warmup:
                lr_scale = min(1., float(total_updates + 1) / float(warmup))
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = lr_scale * lr_real

            loss_opt = res_dict['swav_loss'] + res_dict['mi_loss'] + res_dict['lgt_reg']
            optimizer.zero_grad()
            loss_opt.backward()

            # Stop gradient for prototypes till warmup is over
            if total_updates < warmup:
                model.prototypes.prototypes.weight.grad = None
            optimizer.step()

            epoch_stats.update_dict({'swav_loss': res_dict['swav_loss'].item(), }, n=1)

            # None can be only on STL10, if not enough labelled training instances to evaluate
            if res_dict['y'] is not None:
                predictions.append(res_dict['y'].cpu().numpy())
                epoch_stats.update_dict({
                    'mi_loss': res_dict['mi_loss'].item(),
                    'lgt_reg': res_dict['lgt_reg'].item(),
                }, n=1)
            total_updates += 1
        time_stop = time.time()
        spu = (time_stop - time_epoch)
        print('Epoch {0:d}, {1:.4f} sec/epoch'.format(epoch, spu))
        # update learning rate
        scheduler_inf.step()
        targets, predictions = np.concatenate(targets).ravel(), np.concatenate(predictions).ravel()
        test_model(model, test_loader, device, epoch_stats)
        # Evaluation only for the labelled set (in case of STL10)
        update_train_accuracies(epoch_stats, targets[:predictions.shape[0]], predictions, 'Train Clustering ')
        epoch_str = epoch_stats.pretty_string()
        diag_str = '{0:d}: {1:s}'.format(epoch, epoch_str)
        print(diag_str)
        sys.stdout.flush()
        stat_tracker.record_stats(epoch_stats.averages(epoch, prefix='costs/'))
        checkpointer.update(epoch + 1)


def train_model(model, learning_rate, train_loader, test_loader, nmb_crops, stat_tracker,
                checkpointer, device, warmup, epochs, l2_w):
    mods = [m for m in model.modules_]
    optimizer = optim.Adam([{'params': mod.parameters(), 'lr': learning_rate} for i, mod in enumerate(mods)],
                           betas=(0.8, 0.999), weight_decay=l2_w)

    scheduler = MultiStepLR(optimizer, milestones=[150, 300, 400], gamma=0.4)
    _train(model, optimizer, scheduler, train_loader, test_loader, nmb_crops, stat_tracker,
           checkpointer, device, warmup, epochs)
