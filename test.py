import torch

A = torch.randn(6,256)
B = torch.randn(6,256)

C = torch.bmm(A.view(6, 1, 256), B.view(6, 256, 1))

print(C.shape)