# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:50:07 2021

@author: axmao2-c
"""

""" train and test dataset

author baiyu
"""

# 导入所需的库
import random
import numpy as np
import torch
import platform
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Data_Segm函数用于数据分段处理
# 输入:df_data(数据帧),single(是否单样本),tri(是否三样本)
# 输出:samples_all(分段后的样本),labels_all(对应的标签)
def Data_Segm(df_data, single=True, tri=False):
  # 获取所有不同的segment及其计数
  segments,counts = np.unique(df_data["segment"], return_counts = True)
  samples = []
  labels = []
  # 对每个segment进行处理
  for s in segments:
    data_segment = df_data[df_data['segment'] == s]
    sample_persegm = []
    # 每100个数据点取一个200点长的样本
    for j in range(0,len(data_segment),100):
      # 只取陀螺仪数据(Gx,Gy,Gz)
      temp_sample = data_segment[['Gx','Gy','Gz']].iloc[j:j+200,:].values
      if len(temp_sample) == 200:
        sample_persegm.append(temp_sample)
    samples.append(sample_persegm)
    labels.append(list(set(data_segment['label']))[0])

  samples_all = []
  labels_all = []
  # 根据single和tri参数组织最终的样本
  for i in range(len(labels)):
    if single:  # 单样本模式
      for s in samples[i]:
        samples_all.append([s])
        labels_all.append(labels[i])
    if tri:     # 三样本模式
      for j in range(len(samples[i])):
        if (j+2) < len(samples[i]):
          samples_all.append([samples[i][j], samples[i][j+1], samples[i][j+2]])
          labels_all.append(labels[i])
  
  return samples_all, labels_all

# get_data函数用于获取训练、验证和测试数据
# 输入:train_subjects(训练集主体),valid_subject(验证集主体),test_subject(测试集主体)
# 输出:df_train(训练数据),df_valid(验证数据),df_test(测试数据)
def get_data(train_subjects = [2,3,7,8], valid_subject = 11, test_subject = 14):
    # 根据操作系统选择数据路径
    path = 'E:\\program\\aaa_DL_project\\tsai-0.3.9\\实验从CMI-Net\\data\\Acc_Gyr_Data.csv'
    if platform.system()=='Linux': 
        path = '/home/axmao2/data/equine/Acc_Gyr_Data.csv'
    df_train_raw = pd.read_csv(path)
    df_train_raw = df_train_raw.drop(['sample_index'], axis=1)

    # 将行为标签转换为数值
    # 0:eating/grazing, 1:galloping, 2:standing, 3:trotting, 4:walking-natural, 5:walking-rider
    df_train_raw.replace({'grazing':0,'eating':0,'galloping-natural':1,'galloping-rider':1,'standing':2,'trotting-rider':3,'trotting-natural':3,'walking-natural':4,'walking-rider':5},inplace = True)
    feature_columns = df_train_raw.columns[0:6]
    
    # 对数据进行标准化处理
    for i in feature_columns:
        s_raw = StandardScaler().fit_transform(df_train_raw[i].values.reshape(-1,1))
        df_train_raw[i]  = s_raw.reshape(-1)
        
    # 根据subject划分训练集、验证集和测试集
    df_train = df_train_raw.loc[df_train_raw['subject'].isin(train_subjects)]
    df_valid = df_train_raw[df_train_raw['subject']==valid_subject]
    df_test = df_train_raw[df_train_raw['subject']==test_subject]
    
    return df_train, df_valid, df_test

# 主程序部分
# 1. 获取数据集
df_train, df_valid, df_test = get_data(train_subjects = [14,2,3,7], valid_subject = 8, test_subject = 11)

# 2. 数据分段处理
samples_train, labels_train = Data_Segm(df_train, single=True, tri=False)
samples_valid, labels_valid = Data_Segm(df_valid, single=True, tri=False)
samples_test, labels_test = Data_Segm(df_test, single=True, tri=False)

# 3. 转换为PyTorch张量
tensor_samples_train = torch.from_numpy(np.array(samples_train)).float()
tensor_label_train = torch.from_numpy(np.array(labels_train)).type(torch.LongTensor)

tensor_samples_valid = torch.from_numpy(np.array(samples_valid)).float()
tensor_label_valid = torch.from_numpy(np.array(labels_valid)).type(torch.LongTensor)

tensor_samples_test = torch.from_numpy(np.array(samples_test)).float()
tensor_label_test = torch.from_numpy(np.array(labels_test)).type(torch.LongTensor)

# 4. 保存处理后的数据
torch.save([tensor_samples_train, tensor_samples_valid, tensor_samples_test, tensor_label_train, tensor_label_valid, tensor_label_test], "E:\\program\\aaa_DL_project\\tsai-0.3.9\\实验从CMI-Net\\data\\myTensor_Gyr_6.pt")





