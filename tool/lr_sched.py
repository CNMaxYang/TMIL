# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, max_epochs, warmup_epochs, max_lr, min_lr=1e-5):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        if epoch > max_epochs:
            lr = min_lr
        else:
            lr = min_lr + (max_lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_milestones(optimizer, epoch, warmup_epochs, milestones, gamma, max_lr):
    temp = optimizer.state_dict()['param_groups'][0]['lr']
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
        temp = lr
    else:
        if epoch in milestones:
            temp = temp * gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = temp
    return temp
