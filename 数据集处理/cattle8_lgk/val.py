# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
data = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cattle8_lgk\\cow_activity_data.pt')
print(f"训练集形态：{data['train_data'].shape}")
print(f"标签分布：{np.unique(data['train_labels'].numpy(), return_counts=True)}")