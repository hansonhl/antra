import numpy as np
import copy

import torch
from intervention.location import Location

def copy_helper(x):
    if isinstance(x, (list, tuple, str, dict, np.ndarray)):
        return copy.deepcopy(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().clone()
    else:
        return x


def serialize(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 0:
            return x.item()
        elif len(x.shape) == 1:
            return tuple(x.tolist())
        elif len(x.shape) == 2:
            return tuple(tuple(d0) for d0 in x.tolist())
        elif len(x.shape) == 3:
            return tuple(tuple(tuple(d1) for d1 in d0) for d0 in x.tolist())
        elif len(x.shape) == 4:
            return tuple(tuple(tuple(tuple(d2) for d2 in d1) for d1 in d0) for d0 in x.tolist())
        else:
            raise NotImplementedError(f"cannot serialize x with {len(x.shape)} dimensions")
    elif isinstance(x, np.ndarray):
        return x.tostring()
    else:
        raise ValueError(f"Does not support input type: {type(x)}")

def deserialize(x: tuple):
    return torch.tensor(x)

def stringify_mapping(m):
    res = {}
    for high, low in m.items():
        low_dict = {}
        for low_node, low_loc in low.items():
            if isinstance(low_loc, slice):
                str_low_loc = Location.slice_to_str(low_loc)
            else:
                str_low_loc = str(low_loc)
            low_dict[low_node] = str_low_loc
        res[high] = low_dict
    return res

