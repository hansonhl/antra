from typing import Union, Tuple
import itertools



LocationType = Union[None, int, slice, type(...), Tuple['LocationType', ...]]
SerializedLocationType = Union[None, int, str, Tuple["SerializedLocationType", ...]]

class Location:
    """A helper class to manage parsing of indices and slices"""

    def __init__(self, loc: Union[None, LocationType, str]=None):
        self._loc = Location.process(loc) if loc else None

    def __getitem__(self, item):
        if item is None:
            return (None,)
        return item

    @staticmethod
    def process(x):
        if isinstance(x, (int, tuple, slice)) or x is Ellipsis:
            return x
        elif isinstance(x, str):
            return parse_str(x)
        else:
            raise ValueError(f"Invalid input type {type(x)}. Must be int, tuple, slice, str or Ellipsis")


def parse_str(s):
    if "," not in s:
        return parse_dim(s.strip())
    else:
        return tuple(parse_dim(x.strip()) for x in s.split(","))

def parse_dim(s):
    s = s.strip("[]")
    return Ellipsis if s == "..." \
        else True if s == "True" \
        else False if s == "False" \
        else str_to_slice(s) if ":" in s \
        else int(s)

def str_to_slice(s: str) -> slice:
    return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))

def slice_to_str(s: slice) -> str:
    return ':'.join(x if x != "None" else "" for x in str(s).strip("slice").strip("()").split(", "))

def location_to_str(l: LocationType, add_brackets=False) -> str:
    if isinstance(l, int):
        s = str(l)
    elif isinstance(l, slice):
        s = slice_to_str(l)
    elif l is Ellipsis:
        s = "..."
    elif l is None:
        s = "None"
    elif isinstance(l, tuple):
        s = ",".join(location_to_str(s) for s in l)
    else:
        raise ValueError(f"Invalid input type {type(l)}. Must be int, tuple or slice")

    if add_brackets: s = "[" + s + "]"
    return s

def serialize_location(l: LocationType) -> SerializedLocationType:
    if isinstance(l, int) or l is None:
        return l
    elif isinstance(l, slice):
        return slice_to_str(l)
    elif l is Ellipsis:
        return "..."
    else:
        return tuple(serialize_location(x) for x in l)


def deserialize_location(s: SerializedLocationType) -> LocationType:
    if isinstance(s, int) or s is None:
        return s
    elif isinstance(s, str):
        if s == "...": return  Ellipsis
        else: return str_to_slice(s)
    if isinstance(s, tuple):
        return tuple(deserialize_location(x) for x in s)


_LOC = Location()
full_slice = _LOC[:]

# TODO: Test this
def reduce_dim(l: LocationType, dim) -> LocationType:
    if l is None or l is Ellipsis: return l
    elif isinstance(l, int):
        if dim == 0: raise ValueError(f"cannot reduce Location {location_to_str(l, add_brackets=True)} which is not full at dim 0")
        else: return l
    else: # l is tuple
        if dim >= len(l) or l[dim] is Ellipsis: return l
        if l[dim] != full_slice:
            raise ValueError(f"cannot reduce Location {location_to_str(l, add_brackets=True)} which is not full at dim 0")
        return l[:dim] + l[dim+1:] if dim != -1 else l[:-1]



def expand_dim(l: LocationType, dim) -> LocationType:
    if not isinstance(l, tuple):
        l = (l,)
    if not -len(l) - 1 <= dim <= len(l):
        raise ValueError(f"Cannot expand {len(l)}-dim index at dim {dim} (must be <= {len(l)})")
    dim = dim % (len(l) + 1)
    return l[:dim] + (full_slice,) + l[dim:]

def generate_all_locations(length, batched=True):#(length: int) -> set(LocationType):
    locations = []
    for boolean_assignment in itertools.product([True, False], repeat=length):
        starts = []
        ends  =  []
        is_start = True
        for i in range(length):
            if boolean_assignment[i] and is_start:
                starts.append(i)
                is_start = False
            elif not boolean_assignment[i] and not is_start:
                ends.append(i)
                is_start = True
            if boolean_assignment[i] and i == length - 1 and not is_start:
                ends.append(length)
        if batched:
            if len(starts) == 1 and len(ends) == 1:
                locations.append(_LOC[:,starts[0]:ends[0]])
            else:
                locations.append([_LOC[:,start:end] for start,end in zip(starts,ends)] )
        else:
            if len(starts) == 1 and len(ends) == 1:
                locations.append(_LOC[starts[0]:ends[0]])
            else:
                locations.append([_LOC[start:end] for start,end in zip(starts,ends)] )
    return locations
