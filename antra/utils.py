import sys
import copy

from .location import Location
from .location import slice_to_str
from typing import *

_LOC = Location()

# type aliases
SerializedType = Union[bool, int, float, str, frozenset, bytes, complex, None, Tuple["SerializableType", ...]]

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

def is_numpy_scalar(x):
    if 'numpy' in sys.modules:
        return sys.modules['numpy'].isscalar(x)
    else:
        return False

def is_serialized(x):
    return isinstance(x, (bool, int, float, str, frozenset, bytes, complex)) or x is None or isinstance(x, tuple) and all(is_serialized(y) for y in x)

def copy_helper(x):
    if isinstance(x, (list, tuple, str, dict)) or is_numpy_array(x):
        return copy.deepcopy(x)
    elif is_torch_tensor(x):
        return x.detach().clone()
    else:
        return x

def serialize(x):
    """ Serialize x so that it is hashable. Recursively converts everything into tuples."""
    if isinstance(x, (bool, int, float, str, frozenset, bytes, complex)) or x is None:
        return x
    elif is_numpy_scalar(x):
        return x
    elif is_torch_tensor(x) or is_numpy_array(x):
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
            return tuple(serialize(z) for z in x.tolist())
    # elif is_numpy_array(x):
    #     return x.tostring()
    elif isinstance(x, (tuple, list)):
        return tuple(serialize(z) for z in x)
    elif isinstance(x, set):
        return tuple(sorted(serialize(z)) for z in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, serialize(z)) for k, z in x.items()))
    else:
        raise ValueError(f"Does not support input type: {type(x)}")

def _split_np_array(a, dim):
    if 'numpy' in sys.modules and is_numpy_array(a):
        return sys.modules['numpy'].moveaxis(a, dim, 0)
    else:
        raise ValueError(f"Batched GraphInput values must be np.ndarray or torch.tensor. Given: {type(a)}")

def serialize_batch(d: Dict[str, Any], dim: int) -> List[Tuple]:
    """ Serialize a dict of batched inputs.

    :param d: Dictionary mapping from node names to values batched into a
        torch tensor or numpy array
    :param dim: Batch dimension
    :return: A list containing a hashable key for each input value in the batch
    """
    split_values = [v.unbind(dim=dim) if is_torch_tensor(v) else _split_np_array(v, dim) for v in d.values()]
    return [tuple(sorted((k, serialize(a)) for k, a in zip(d.keys(), x)))
            for x in zip(*split_values)]


def idx_by_dim(x: Any, idx: int, dim: int):
    """Index into a tensor at a given dimension
    e.g. x = torch.tensor([[2,3,4], [13,14,15]])
    idx_by_dim(x, 1, 1) returns [3, 14]
    idx_by_dim(x, 0, 1) returns [13, 14, 15]
    """
    if is_torch_tensor(x) or is_numpy_array(x):
        if dim == 0:
            return x[idx]
        else:
            loc = tuple([_LOC[:]] * dim + [idx])
            return x[loc]
    else:
        raise ValueError

def stringify_mapping(m):
    res = {}
    for high, low in m.items():
        low_dict = {}
        for low_node, low_loc in low.items():
            if isinstance(low_loc, slice):
                str_low_loc = slice_to_str(low_loc)
            else:
                str_low_loc = str(low_loc)
            low_dict[low_node] = str_low_loc
        res[high] = low_dict
    return res

