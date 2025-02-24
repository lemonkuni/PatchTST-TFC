import torch

# Load the .pt file
# tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\25Hz_data.pt')

tensors = torch.load('E:\\program\\aaa_DL_project\\0000PatchTST-TFC\\CMI-Net\\data\\myTensor_Gyr_6.pt')

# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')
