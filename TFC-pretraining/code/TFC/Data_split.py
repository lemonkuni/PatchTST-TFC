"""This file aims to generate a different data split to evaluate the stability of models.
In each split, we select 30 positive and 30 negative samples to form a training set (60 samples in total).
This is an example on Epilepsy dataset. -- Xiang Zhang, Jan 16, 2023"""

"""
这段代码的主要功能是重新划分Epilepsy数据集,生成新的训练集和测试集。主要步骤如下:

1. 加载数据:
- 从Epilepsy目录下加载已有的训练集和测试集数据
- 将两个数据集的样本和标签分别合并到一起

2. 生成平衡的训练集:
- 设定训练集大小为30个样本
- 分别找出标签为0和1的样本
- 从每类中取30个样本组成训练集(共60个样本)
- 剩余样本组成测试集

3. 保存新的数据集划分:
- 将新生成的训练集和测试集分别保存为train.pt和test.pt
- 保存到原来的Epilepsy数据集目录下

注释掉的代码是生成不平衡训练集的方法,通过随机打乱来选择训练样本。
"""

import torch
import os
import numpy as np

# 设置目标数据集路径
targetdata_path = f"../../datasets/Epilepsy/"

# 加载现有的数据集
finetune_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
train_data = torch.load(os.path.join(targetdata_path, "train.pt"))

# 合并所有样本和标签
Samples = torch.cat((finetune_dataset['samples'], train_data['samples']), dim=0)
Labels = torch.cat((finetune_dataset['labels'], train_data['labels']), dim=0)

# 设置每类训练样本数量
train_size = 30

# 生成平衡的训练集
id0 = Labels==0  # 找出标签为0的样本索引
id1 = Labels==1  # 找出标签为1的样本索引

# 按标签分类样本
Samples_0, Samples_1 = Samples[id0], Samples[id1]
Labels_0, Labels_1 = Labels[id0], Labels[id1]

# 构建训练集和测试集
Samples_train = torch.cat((Samples_0[:train_size], Samples_1[:train_size]))  # 每类取30个样本作为训练集
Samples_test = torch.cat((Samples_0[train_size:], Samples_1[train_size:]))   # 剩余样本作为测试集

Labels_train = torch.cat((Labels_0[:train_size], Labels_1[:train_size]))     # 训练集标签
Labels_test = torch.cat((Labels_0[train_size:], Labels_1[train_size:]))      # 测试集标签

# 保存新的数据集划分
train_dic = {'samples':Samples_train, 'labels':Labels_train}
test_dic = {'samples':Samples_test, 'labels':Labels_test}
torch.save(train_dic, os.path.join(targetdata_path,'train.pt'))
torch.save(test_dic, os.path.join(targetdata_path,'test.pt'))

print('Re-split finished. Dataset saved to folder:', targetdata_path)
