# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
data = torch.load('cow_activity_data.pt')
print(f"训练集形态：{data['train_data'].shape}")
print(f"标签分布：{np.unique(data['train_labels'].numpy(), return_counts=True)}")