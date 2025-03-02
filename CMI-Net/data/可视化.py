import torch

# Load the .pt file
# tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\25Hz_data.pt')

# tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\myTensor_Gyr_6.pt')

tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\new_goat_25hz_3axis.pt')

# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')


# 查看第四个张量（标签）的唯一值
labels = tensors[3]  # 第四个元素是训练集的标签
unique_labels = torch.unique(labels)
print("\n训练集标签的唯一值：")
print(unique_labels)

# 统计每个标签的数量
for label in unique_labels:
    count = (labels == label).sum().item()
    print(f"标签 {label.item()} 的数量: {count}")

for i in range(10):
    print(tensors[0][i+10000],tensors[3][i+10000]);
