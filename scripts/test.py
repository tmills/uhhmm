import torch

a = torch.zeros(4, 1372, 32)
b = torch.zeros(4, 1372, 32)
a = a.cuda()

a.fill_(0)