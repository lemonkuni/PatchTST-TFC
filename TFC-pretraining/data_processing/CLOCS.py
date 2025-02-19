"""
https://arxiv.org/abs/2005.13249
这段代码的主要功能是将不同数据集转换为CLOCS模型所需的格式并保存。具体来说:

1. 定义了8个数据集的相关信息:
- alias_lst: 数据集别名
- dirname_lst: 数据集目录名
- trial_lst: 实验类型(contrastive_ms或contrastive_ss)
- modality_lst: 数据模态(eeg/ecg/emg/other)

2. 对每个数据集进行处理:
- 加载train/val/test三个数据集的pt文件
- 创建三个嵌套字典结构:
  - input_dict: 存储输入数据
  - output_dict: 存储标签
  - pid_dict: 存储样本ID

3. 数据处理:
- 从pt文件中提取samples作为输入数据
- 提取labels作为输出标签
- 为每个样本生成唯一的ID

4. 保存处理后的数据:
- 将三个字典分别保存为pkl文件:
  - frames_phases_{alias}.pkl: 输入数据
  - labels_phases_{alias}.pkl: 标签
  - pid_phases_{alias}.pkl: 样本ID
- 保存路径为CLOCS模型的数据目录

这样处理后的数据可以直接用于CLOCS模型的训练。
"""

import torch
import numpy as np
import os
import pickle

alias_lst = ['sleepEDF', 'epilepsy', 'pFD_A', 'pFD_B', 'HAR', 'AHAR', 'physionet2017', 'emg']
dirname_lst = ['SleepEEG', 'Epilepsy', 'FD-A', 'FD-B', 'HAR', 'Gesture', 'ecg', 'emg']
trial_lst = ['contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss']
phase_lst = ['train','val','test']
modality_lst = ['eeg', 'eeg', 'other', 'other', 'other', 'other', 'ecg', 'emg']
fraction = 1
term = 'All Terms'
desired_leads = ['I']

for alias, dirname, trial, modality in zip(alias_lst, dirname_lst, trial_lst, modality_lst):
    train_dict = torch.load(os.path.join('datasets', dirname, 'train.pt'))
    val_dict = torch.load(os.path.join('datasets', dirname, 'val.pt'))
    test_dict = torch.load(os.path.join('datasets', dirname, 'test.pt'))

    input_dict = {}
    output_dict = {}
    pid_dict = {}
    input_dict[modality] = {}
    output_dict[modality] = {}
    pid_dict[modality] = {}
    input_dict[modality][fraction] = {}
    output_dict[modality][fraction] = {}
    pid_dict[modality][fraction] = {}

    for phase in phase_lst:
        input_dict[modality][fraction][phase] = {}
        output_dict[modality][fraction][phase] = {}
        pid_dict[modality][fraction][phase] = {}

    input_dict[modality][fraction]['train'][term] = train_dict['samples'][:,0,:]
    input_dict[modality][fraction]['test'][term] = test_dict['samples'][:,0,:]
    input_dict[modality][fraction]['val'][term] = val_dict['samples'][:,0,:]

    output_dict[modality][fraction]['train'][term] = np.expand_dims(train_dict['labels'], axis=1)
    output_dict[modality][fraction]['test'][term] = np.expand_dims(test_dict['labels'], axis=1)
    output_dict[modality][fraction]['val'][term] = np.expand_dims(val_dict['labels'], axis=1)

    ctr = 0
    pid_dict[modality][fraction]['train'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(train_dict['labels'])), axis=1)
    ctr += len(train_dict['labels'])
    pid_dict[modality][fraction]['test'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(test_dict['labels'])), axis=1)
    ctr += len(test_dict['labels'])
    pid_dict[modality][fraction]['val'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(val_dict['labels'])), axis=1)

    savepath = os.path.join('code', 'baselines', 'CLOCS', 'data', alias, trial,'leads_%s' % str(desired_leads))
    if os.path.isdir(savepath) == False:
        os.makedirs(savepath)

    """ Save Frames and Labels Dicts """
    with open(os.path.join(savepath,'frames_phases_%s.pkl' % alias),'wb') as f:
        pickle.dump(input_dict,f)

    with open(os.path.join(savepath,'labels_phases_%s.pkl' % alias),'wb') as g:
        pickle.dump(output_dict,g)

    with open(os.path.join(savepath,'pid_phases_%s.pkl' % alias),'wb') as h:
        pickle.dump(pid_dict,h)

    print(f'Final Frames Saved for {alias}!')
