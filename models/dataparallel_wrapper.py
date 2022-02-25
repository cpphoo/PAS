import torch.nn as nn


class dataparallel_wrapper(nn.Module):
    '''
        A wrapper for nn.DataParallel

        Wrapping a torch module with this wrapper allow the the forward function 
        for nn.DataParallel to call other methods of the wrapped module 
    '''
    def __init__(self, module):
        super(dataparallel_wrapper, self).__init__()
        self.module = module

    def forward(self, mode='forward', *args, **kwargs):
        return getattr(self.module, mode)(*args, **kwargs)
