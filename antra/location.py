from typing import Union, Tuple

LocationType = Union[int, slice, type(...), Tuple["LocationType", ...]]
SerializedLocationType = Union[int, str, Tuple["SerializedLocationType", ...]]

class Location:
    """A helper class to manage parsing of indices and slices"""

    def __init__(self, loc: Union[None, LocationType, str]=None):
        self._loc = Location.process(loc) if loc else None

    def __getitem__(self, item):
        return item

    @staticmethod
    def process(x):
        if isinstance(x, (int, tuple, slice)) or x is Ellipsis:
            return x
        elif isinstance(x, str):
            return Location.parse_str(x)
        else:
            raise ValueError(f"Invalid input type {type(x)}. Must be int, tuple, slice, str or Ellipsis")

    @staticmethod
    def parse_str(s):
        if "," not in s:
            return Location.parse_dim(s.strip())
        else:
            return tuple(Location.parse_dim(x.strip()) for x in s.split(","))

    @staticmethod
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
    elif isinstance(l, tuple):
        s = ",".join(location_to_str(s) for s in l)
    else:
        raise ValueError(f"Invalid input type {type(l)}. Must be int, tuple or slice")

    if add_brackets: s = "[" + s + "]"
    return s

def serialize_location(l: LocationType) -> SerializedLocationType:
    if isinstance(l, int):
        return l
    elif isinstance(l, slice):
        return slice_to_str(l)
    elif l is Ellipsis:
        return "..."
    else:
        return tuple(serialize_location(x) for x in l)


def deserialize_location(s: SerializedLocationType) -> LocationType:
    if isinstance(s, int):
        return s
    elif isinstance(s, str):
        if s == "...": return  Ellipsis
        else: return str_to_slice(s)
    if isinstance(s, tuple):
        return tuple(deserialize_location(s))


