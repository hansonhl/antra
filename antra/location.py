class Location:
    """A helper class to manage parsing of indices and slices"""

    def __init__(self, loc=None):
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
            else Location.str_to_slice(s) if ":" in s \
            else int(s)

    @staticmethod
    def str_to_slice(s):
        return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))

    @staticmethod
    def slice_to_str(s):
        return ':'.join(x if x != "None" else "" for x in str(s).strip("slice").strip("()").split(", "))

    @staticmethod
    def loc_to_str(l, add_brackets=False):
        if isinstance(l, tuple):
            s = ",".join(Location.slice_to_str(s) for s in l)
        elif isinstance(l, slice):
            s = Location.slice_to_str(l)
        elif isinstance(l, int):
            s = str(l)
        else:
            raise ValueError(f"Invalid input type {type(l)}. Must be int, tuple or slice")

        if add_brackets: s = f"[{s}]"
        return s


