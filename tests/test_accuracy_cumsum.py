import torch

device='cuda'
N = 100000000 #100 million
divs = 10000000
delta = 10
verylongtensor = torch.ones(N, device=device, dtype=torch.float32)
cumsumtensor = torch.cumsum(verylongtensor, 0)

print(cumsumtensor[-1]-cumsumtensor[-1-delta])

#compared to other option; scatter_add
addlen = delta
scatter_ind = torch.arange(divs).repeat_interleave(addlen)
scatter_result = torch.zeros(divs).scatter_add(0, scatter_ind, verylongtensor)

print(scatter_result[-1])
