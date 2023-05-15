import torch

#This interface provides functionality which should make mapping data to gpu possible.
class Torch_Struct:

    def __init__(self):
        self._data = torch.tensor() #everything should be stored in torch tensors, to allow for easy mapping
    # Computes the size of in bytes of the structure (for correctly allocating gpu memory)
    def get_total_size_bytes(self):
        return self.get_element_size_bytes()*self.get_total_size()
    def get_element_size_bytes(self):
        return self._data.element_size()
    # Number of individual data points
    def get_total_size(self):
        return self._data.nelement()
    # Hmm, maybe also add option of size per index; or just get number of 
    # Maps the entire vector to the accelerator; TODO: also allow mapping a part to accelerator in subclasses
    # TODO: also allow choosing which accelerator to map to
    def map_to_accelerator(self):
        pass
    # Maps everything currently mapped to accelerator back to cpu; TODO: also allow mapping a part of the indices back
    def map_to_cpu(self):
        pass
    # Depending on the data structure, we also need to define access functions for the data; these will not be for use during actual computations, but only for inspecting the data after computing

