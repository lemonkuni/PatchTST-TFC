import torch

# 加载原始pt文件
original_data = torch.load("new_goat_myTensor_1.pt")
tensor_samples_train, tensor_samples_valid, tensor_samples_test, tensor_label_train, tensor_label_valid, tensor_label_test = original_data

def process_samples(samples_tensor):
    """处理样本数据：选择三轴 + 降采样"""
    # 输入形状: (num_samples, 1, 200, 36)
    # 提取axA/ayA/azA三轴（前三列）
    selected_axes = samples_tensor[:, :, :, :3]
    # 时间维度降采样（每4个取1个）
    downsampled = selected_axes[:, :, ::4, :]
    return downsampled

# 处理所有样本集
new_samples_train = process_samples(tensor_samples_train)
new_samples_valid = process_samples(tensor_samples_valid)
new_samples_test = process_samples(tensor_samples_test)

# 验证形状 (示例检查)
print("原始训练样本形状:", tensor_samples_train.shape)  # 应为 (N, 1, 200, 36)
print("新训练样本形状:", new_samples_train.shape)     # 应为 (N, 1, 50, 3)

# 保存新的pt文件（保持原有数据结构）
torch.save([
    new_samples_train,
    new_samples_valid,
    new_samples_test,
    tensor_label_train,
    tensor_label_valid,
    tensor_label_test
], "new_goat_25hz_3axis.pt")