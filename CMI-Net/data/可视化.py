import torch

# Load the .pt file
tensors = torch.load('E:\\program\\aaa_DL_project\\tsai-0.3.9\\实验从CMI-Net\\data\\myTensor_Acc_25Hz_datasets.pt')

# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')
