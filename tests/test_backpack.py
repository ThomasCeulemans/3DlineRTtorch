import torch
from backpack import extend, backpack

N = 10
device = 'cpu'
data = torch.rand(N, device=device, requires_grad=True)
otherdata = torch.rand(N, device=device)
overwritabledat = data
for i in range(2):
    overwritabledat = overwritabledat*otherdata

overwritabledat.backward(gradient=torch.ones(N))
print(data, data.grad, otherdata**2)