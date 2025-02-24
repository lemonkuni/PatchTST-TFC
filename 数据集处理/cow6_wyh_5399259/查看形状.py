import torch

# Load the .pt file
tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\数据集处理\\cow6_wyh_5399259\\cow6_wyh_dataset.pt')

# Print the type and shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} type: {type(tensor)}')
    if isinstance(tensor, torch.Tensor):
        print(f'Tensor {i+1} shape: {tensor.shape}')
    else:
        print(f'Tensor {i+1} is not a tensor, it is a {type(tensor)}')