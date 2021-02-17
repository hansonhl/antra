import sys
import copy

if 'torch' in sys.modules:
    pass
if 'numpy' in sys.modules:
    pass
from .location import Location

def is_torch_tensor(x):
    if 'torch' in sys.modules:
        return isinstance(x, sys.modules['torch'].Tensor)
    else:
        return False

def is_numpy_array(x):
    if 'numpy' in sys.modules:
        return isinstance(x, sys.modules['numpy'].ndarray)
    else:
        return False

def copy_helper(x):
    if isinstance(x, (list, tuple, str, dict)) or is_numpy_array(x):
        return copy.deepcopy(x)
    elif is_torch_tensor(x):
        return x.detach().clone()
    else:
        return x

def serialize(x):
    if isinstance(x, (bool, int, float, str, frozenset, bytes, complex)):
        return x
    elif is_torch_tensor(x):
        if len(x.shape) == 0:
            return x.item()
        elif len(x.shape) == 1:
            return tuple(x.tolist())
        elif len(x.shape) == 2:
            return tuple(tuple(d1) for d1 in x.tolist())
        elif len(x.shape) == 3:
            return tuple(tuple(tuple(d2) for d2 in d1) for d1 in x.tolist())
        elif len(x.shape) == 4:
            return tuple(tuple(tuple(tuple(d3) for d3 in d2) for d2 in d1) for d1 in x.tolist())
        elif len(x.shape) == 5:
            return tuple(tuple(tuple(tuple(tuple(d4) for d4 in d3) for d3 in d2) for d2 in d1) for d1 in x.tolist())
        else:
            raise NotImplementedError(f"cannot serialize x with {len(x.shape)} dimensions")
    elif is_numpy_array(x):
        return x.tostring()
    elif isinstance(x, (tuple, list)):
        return tuple(serialize(z) for z in x)
    elif isinstance(x, set):
        return tuple(sorted(serialize(z)) for z in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, serialize(z)) for k, z in x.items()))
    else:
        raise ValueError(f"Does not support input type: {type(x)}")


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
