import torch
from . import torch_struct

class Torch_Vector(Torch_Struct):

    def __init__(self):
        self._data = torch.tensor()
    def __init__(self, tensor : torch.Tensor, autograd = False):
        shape = tensor.shape()
        assert len(shape) == 1 #assert 1D tensor as input
        self._data = torch.tensor(tensor, autograd = autograd)
        
    
