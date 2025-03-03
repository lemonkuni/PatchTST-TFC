# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:37:49 2021

@author: axmao2-c
"""

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
   
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F



def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
  
  
    """计算类别平衡损失(Class Balanced Loss)。
    
    该损失函数通过重新加权每个类别的损失来处理类别不平衡问题。权重基于每个类别的样本数量,使用公式:
    Class Balanced Loss = ((1-beta)/(1-beta^n)) * Loss(labels, logits)
    其中 n 是每个类别的样本数量。

    使用方法:
    >>> # 示例:
    >>> labels = torch.tensor([0, 1, 2, 1])  # 批次中的类别标签
    >>> logits = torch.randn(4, 3)  # 模型输出的预测分数,shape为[batch_size, num_classes] 
    >>> samples_per_cls = [100, 50, 25]  # 每个类别的样本总数
    >>> no_of_classes = 3  # 类别总数
    >>> loss_type = "focal"  # 可选: "focal", "sigmoid", "softmax"
    >>> beta = 0.9999  # 类别平衡的超参数,通常接近1
    >>> gamma = 2.0  # Focal Loss的聚焦参数
    >>> loss = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)

    参数:
        labels (torch.Tensor): 形状为[batch]的整数张量,包含每个样本的类别标签。
        logits (torch.Tensor): 形状为[batch, no_of_classes]的浮点张量,包含模型的原始输出分数。
        samples_per_cls (list/torch.Tensor): 大小为[no_of_classes]的列表,包含每个类别的样本总数。
        no_of_classes (int): 类别总数。
        loss_type (str): 使用的基础损失函数类型,可选"sigmoid"、"focal"或"softmax"。
        beta (float): 类别平衡的超参数,用于计算有效样本数,通常接近1。
        gamma (float): Focal Loss的聚焦参数,仅在loss_type="focal"时使用。

    返回:
        torch.Tensor: 标量张量,表示计算得到的类别平衡损失值。
    """

    
    device = logits.device
    labels = labels.to(device)
    
    if not isinstance(samples_per_cls, torch.Tensor):
        samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float)
    samples_per_cls = samples_per_cls.to(device)

    # 检查标签值是否在有效范围内
    if labels.max() >= no_of_classes:
        raise ValueError(
            f"Labels contain invalid class index. Max label: {labels.max()}, "
            f"num_classes: {no_of_classes}, unique labels: {torch.unique(labels)}"
        )
    
    # 计算权重############################################################################
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * no_of_classes
    weights = weights.to(device)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    ########################################################################################
    
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss
