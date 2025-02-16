import torch

# Load the .pt file
tensors = torch.load('dataset.pt')

# Print the shape of each tensor
for i, tensor in enumerate(tensors):
    print(f'Tensor {i+1} shape: {tensor.shape}')