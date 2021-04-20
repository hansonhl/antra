from typing import *

import antra.utils as utils
from antra.utils import SerializedType
from antra.location import SerializedLocationType

SerializedRealization=Tuple[Tuple[Tuple[str, SerializedLocationType], SerializedType], ...]
RealizationKey = Tuple[str, SerializedLocationType]
Intervention = "antra.intervention.Intervention"

class Realization:
    """ An object that is a 'duck typing' of a dict that maps node names and
    locations to values. Also optionally keeps a reference to an Intervention that
    produced that value. """
    def __init__(self):
        self._data: Dict[RealizationKey, Any] = {}
        self._origins: Dict[RealizationKey, Intervention] = {}
        self.accepted = False
        
    @classmethod
    def deserialize(cls, ser_rzn: SerializedRealization):
        pass
        
    @property
    def data(self):
        return self._data
    
    @property
    def origins(self):
        return self._origins

    def add(self, key: RealizationKey, value: Any, origin: Intervention):
        self._data[key] = value
        self._origins[key] = origin

    def is_empty(self):
        return len(self._data) == 0

    def __getitem__(self, item) -> Any:
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __copy__(self):
        # shallow copy called by copy.copy()
        new_rzn = Realization()
        new_rzn.update(self)
        return new_rzn
        
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def values(self):
        return self._data.values()
        
    def update(self, other):
        for k, v in other.items():
            self._data[k] = v
        for k, v in other.origins.items():
            self._origins[k] = v

    def serialize(self) -> SerializedRealization:
        return tuple((k, utils.serialize(self[k])) for k in sorted(self.keys()))

    def __repr__(self):
        repr_dict = {
            "data": self.data,
            "origins": self.origins
        }