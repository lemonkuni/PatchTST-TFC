# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:50:07 2021

@author: axmao2-c
"""

""" train and test dataset

author baiyu
"""

import random
import numpy as np
import torch

import platform
import pandas as pd
from sklearn.preprocessing import StandardScaler


#Data segmentation
def Data_Segm(df_data, single=True, tri=False):
  segments,counts = np.unique(df_data["segment"], return_counts = True)
  samples = []
  labels = []
  for s in segments:
    data_segment = df_data[df_data['segment'] == s]
    sample_persegm = []
    for j in range(0,len(data_segment),100):
      # temp_sample = data_segment[['Ax','Ay','Az','Gx','Gy','Gz']].iloc[j:j+200,:].values
      temp_sample = data_segment[['Gx','Gy','Gz']].iloc[j:j+200,:].values
      if len(temp_sample) == 200:
        sample_persegm.append(temp_sample)
    samples.append(sample_persegm)
    labels.append(list(set(data_segment['label']))[0])

  samples_all = []
  labels_all = []
  for i in range(len(labels)):
    if single:
      for s in samples[i]:
        samples_all.append([s])
        labels_all.append(labels[i])
    if tri:
      for j in range(len(samples[i])):
        if (j+2) < len(samples[i]):
          samples_all.append([samples[i][j], samples[i][j+1], samples[i][j+2]])
          labels_all.append(labels[i])
  
  return samples_all, labels_all

#Get training data, validation data, and test data
def get_data(train_subjects = [2,3,7,8], valid_subject = 11, test_subject = 14):
    
    path = 'C:\\Workplace\\Data\\Acc_Gyr_Data.csv'
    if platform.system()=='Linux': 
        path = '/home/axmao2/data/equine/Acc_Gyr_Data.csv'
    df_train_raw = pd.read_csv(path)
    df_train_raw = df_train_raw.drop(['sample_index'], axis=1)

    #数值对应6中行为['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
    df_train_raw.replace({'grazing':0,'eating':0,'galloping-natural':1,'galloping-rider':1,'standing':2,'trotting-rider':3,'trotting-natural':3,'walking-natural':4,'walking-rider':5},inplace = True)
    #class_labels = ['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
    feature_columns = df_train_raw.columns[0:6]
    
    #data standardization2
    for i in feature_columns:
        s_raw = StandardScaler().fit_transform(df_train_raw[i].values.reshape(-1,1))
        df_train_raw[i]  = s_raw.reshape(-1)
        
    #get the training data, validation data, and test data
    df_train = df_train_raw.loc[df_train_raw['subject'].isin(train_subjects)]
    df_valid = df_train_raw[df_train_raw['subject']==valid_subject]
    df_test = df_train_raw[df_train_raw['subject']==test_subject]
    
    return df_train, df_valid, df_test


df_train, df_valid, df_test = get_data(train_subjects = [14,2,3,7], valid_subject = 8, test_subject = 11)
samples_train, labels_train = Data_Segm(df_train, single=True, tri=False)
samples_valid, labels_valid = Data_Segm(df_valid, single=True, tri=False)
samples_test, labels_test = Data_Segm(df_test, single=True, tri=False)

tensor_samples_train = torch.from_numpy(np.array(samples_train)).float()
tensor_label_train = torch.from_numpy(np.array(labels_train)).type(torch.LongTensor)

tensor_samples_valid = torch.from_numpy(np.array(samples_valid)).float()
tensor_label_valid = torch.from_numpy(np.array(labels_valid)).type(torch.LongTensor)

tensor_samples_test = torch.from_numpy(np.array(samples_test)).float()
tensor_label_test = torch.from_numpy(np.array(labels_test)).type(torch.LongTensor)

torch.save([tensor_samples_train, tensor_samples_valid, tensor_samples_test, tensor_label_train, tensor_label_valid, tensor_label_test], "./myTensor_Gyr_6.pt")





