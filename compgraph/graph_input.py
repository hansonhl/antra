from typing import Dict, Any, Sequence

class GraphInput:
    """ A hashable input object that stores a dict mapping names of nodes to
    values of arbitrary type.

    `GraphInput` objects are intended to be immutable, so that its hash value
    can have a one-to-one correspondence to the dict stored in it. """

    def __init__(self, values: Dict[str,Any], cache_results: bool=True,
                 batched: bool=False, batch_dim: int=0, keys: Sequence=None):
        """
        :param values: A dict mapping from each leaf node name (str) to an input
            value for that node (Any)
        :param cache_results: If true then cache the results during computation
        :param batched: If true then indicates `values` contains a batch of
            inputs, i.e. the value of the dict must be a sequence.
        :param batch_dim: If inputs are batched and are pytorch tensors, the
            dimension for the batch
        :param keys: A unique key/hash value for each input value in the batch
        """
        self._values = values
        self.cache_results=cache_results
        self.batched = batched
        self.batch_dim = batch_dim
        self.keys = keys
        if batched and not self.keys:
            raise ValueError("Must provide keys for each element of the batch!")

        # self._all_tensors = len(values) > 0 and all(isinstance(v, torch.Tensor)
        #                                             for v in values.values())
        #
        # # automatically get the device if all input values are tensors and on the same device
        # self.device = None
        # if self._all_tensors:
        #     devices = set(v.device for v in self.values.values())
        #     if len(devices) == 1:
        #         self.device = devices.pop()
        #     else:
        #         raise RuntimeError("Currently does not support input values on multiple devices")

    @classmethod
    def batched(cls, values: Dict[str,Any], keys, cache_results: bool=True,
                 batch_dim: int=0):
        return cls(values, cache_results=cache_results, batched=True,
                   batch_dim=batch_dim, keys=keys)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        raise RuntimeError("GraphInput objects are immutable!")

    def __getitem__(self, item):
        return self.values[item]

    def __contains__(self, item):
        """ Override the python `in` operator """
        return item in self.values

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        if self.values is None:
            return "GraphInput{}"
        else:
            s = ", ".join(
                ("'%s': %s" % (k, type(v))) for k, v in self.values.items())
            return "GraphInput{%s}" % s

    def to(self, device):
        """Move all data to a pytorch Device.

        This does NOT modify the original GraphInput object but returns a new
        one. """
        # assert all(isinstance(t, torch.Tensor) for _, t in self._values.items())

        new_values = {k: v.to(device) for k, v in self._values.items()}
        return GraphInput(new_values)
