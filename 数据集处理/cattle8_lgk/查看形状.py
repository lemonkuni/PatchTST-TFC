import torch

# Load the .pt file
tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cattle8_lgk\\cow_activity_data.pt')

# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')

# Tensor 1 shape: torch.Size([67, 50, 3])
# Tensor 2 shape: torch.Size([10, 50, 3])
# Tensor 3 shape: torch.Size([51, 50, 3])
# Tensor 4 shape: torch.Size([67])
# Tensor 5 shape: torch.Size([10])
# Tensor 6 shape: torch.Size([51])