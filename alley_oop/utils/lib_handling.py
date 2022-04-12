
import numpy
import torch

from typing import Union, Type
from modulefinder import Module

def get_lib(
        data_object: Union[numpy.ndarray, torch.Tensor]
    ) -> Module:

    if isinstance(data_object, numpy.ndarray):
        return numpy
    
    if isinstance(data_object, torch.Tensor):
        return torch

    raise TypeError('%s is not supported' % type(data_object))

def get_class(
        data_object: Union[numpy.ndarray, torch.Tensor]
    ) -> Union[numpy.array, torch.tensor]:

    if isinstance(data_object, numpy.ndarray):
        return numpy.array

    if isinstance(data_object, torch.Tensor):
        return torch.tensor

    raise TypeError('%s is not supported' % type(data_object))    