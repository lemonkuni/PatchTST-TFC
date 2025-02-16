# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def create_segments(df, label_mapping, window_size=50, step=25):
    """
    :param window_size: 窗口大小（25Hz下2秒数据）
    :param step: 滑动步长（50%重叠）
    """
    segments = []
    labels = []

    # 按动物分组处理
    for animal_id, animal_df in df.groupby('animal'):
        # 按datapoint排序确保时序
        sorted_df = animal_df.sort_values('datapoint')
        
        # 降采样：每两个点取一个点
        downsampled_df = sorted_df.iloc[::2].reset_index(drop=True)

        # 提取加速度数据和标签序列
        acc_data = downsampled_df[['mx', 'my', 'mz']].values
        label_seq = downsampled_df['label'].values

        # 滑动窗口分割
        for start in range(0, len(acc_data) - window_size + 1, step):
            end = start + window_size
            window = acc_data[start:end]
            window_labels = label_seq[start:end]

            # 标签一致性要求75%
            unique, counts = np.unique(window_labels, return_counts=True)
            if (counts.max() / window_size) >= 0.75:
                segments.append(window)
                labels.append(unique[np.argmax(counts)])

    return np.array(segments), np.array(labels)


def get_processed_data(data_path, label_mapping):
    # 读取数据
    df = pd.read_csv(data_path)  
    # 直接替换标签
    df.replace({
        'grazing': 0,
        'walking': 1,
        'resting': 2,
        'drinking': 3,
        'alia': 4
    }, inplace=True)

    # 标准化处理
    scaler = StandardScaler()
    acc_columns = ['mx', 'my', 'mz']  # 确认实际加速度列名
    df[acc_columns] = scaler.fit_transform(df[acc_columns])

    # 划分数据集（示例：动物1-5训练，6验证，7-8测试）
    train_df = df[df['animal'].isin([1, 2, 3, 4, 5])]
    valid_df = df[df['animal'] == 6]
    test_df = df[df['animal'].isin([7, 8])]

    # 生成数据片段
    X_train, y_train = create_segments(train_df, label_mapping)
    X_valid, y_valid = create_segments(valid_df, label_mapping)
    X_test, y_test = create_segments(test_df, label_mapping)

    # 转换为张量（形状：样本数 × 时间步长 × 特征数）
    tensor_X_train = torch.FloatTensor(X_train)
    tensor_y_train = torch.LongTensor(y_train)

    tensor_X_valid = torch.FloatTensor(X_valid)
    tensor_y_valid = torch.LongTensor(y_valid)

    tensor_X_test = torch.FloatTensor(X_test)
    tensor_y_test = torch.LongTensor(y_test)

    # 打印一些信息用于调试
    print(f"原始训练数据大小：{len(train_df)}")
    print(f"训练集形态：{tensor_X_train.shape}")
    print(f"训练集标签分布：", np.unique(y_train, return_counts=True))

    # 修改保存格式，与Get_data.py一致
    torch.save([
        tensor_X_train, 
        tensor_X_valid, 
        tensor_X_test, 
        tensor_y_train, 
        tensor_y_valid, 
        tensor_y_test
    ], 'cow_activity_data.pt')



label_mapping = {
    'grazing': 0,
    'walking': 1, 
    'resting': 2,
    'drinking': 3,
    'alia': 4
}

get_processed_data('Arm20c_Accfeatures.csv', label_mapping)