from antra import *
import antra.utils as utils

class Realization:
    def __init__(self):
        self._data = {}
        self._origins = {}
        
    @classmethod
    def deserialize(cls, ser_rzn):
        pass
        
    @property
    def data(self):
        return self._data
    
    @property
    def origins(self):
        return self._origins

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        
    def get_origin(self, key):
        return self._origins[key]
    
    def set_origin(self, key, value):
        self._origins[key] = value
        
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def values(self):
        return self._data.values()
        
    def update(self, other):
        for k, v in other.values:
            self._data[k] = v
        for k, v in other.origins:
            self._origins[k] = v

    def serialize(self):
        return tuple((k, utils.serialize(self[k])) for k in sorted(self.keys()))
    