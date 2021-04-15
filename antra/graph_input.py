import logging
from typing import Dict, Any, Sequence

from .utils import serialize, serialize_batch, is_torch_tensor, is_numpy_array

logger = logging.getLogger(__name__)

class GraphInput:
    """ A hashable input object that stores a dict mapping names of nodes to
    values of arbitrary type.

    `GraphInput` objects are intended to be immutable, so that its hash value
    can have a one-to-one correspondence to the dict stored in it. """

    def __init__(self, values: Dict[str,Any], cache_results: bool=True,
                 batched: bool=False, batch_dim: int=0, keys: Sequence=None,
                 key_leaves: Sequence[str]=None):
        """
        :param values: A dict mapping from each leaf node name (str) to an input
            value for that node (Any)
        :param cache_results: If true then cache the results during computation
        :param batched: If true then indicates `values` contains a batch of
            inputs, i.e. the value of the dict must be a sequence.
        :param batch_dim: If inputs are batched and are pytorch tensors, the
            dimension for the batch
        :param keys: A unique key/hash value for each input value in the batch
        :param key_leaves: Specify a (sub)set of leaves whose values are used
            to automatically calculate the key
        """
        self._values = values
        self.cache_results=cache_results
        self.batched = batched
        self.batch_dim = batch_dim
        self.key_leaves = key_leaves

        if batched and not keys:
            keys = serialize_batch(
                {k: v for k, v in values.items() if k in key_leaves} if key_leaves else values,
                dim=batch_dim
            )
        if not batched and not keys:
            keys = serialize(
                {k: v for k, v in values.items() if k in key_leaves} if key_leaves else values
            )

        self.keys = keys
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
    def batched(cls, values: Dict[str,Any], keys=None, cache_results: bool=True,
                 batch_dim: int=0):
        return cls(values, cache_results=cache_results, batched=True,
                   batch_dim=batch_dim, keys=keys)

    @property
    def values(self) -> Dict[str, Any]:
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

    def is_empty(self):
        return not self._values

    def get_batch_size(self):
        if not self.batched:
            return None
        batch_size = None
        for k, v in self.values.items():
            if self.key_leaves and k not in self.key_leaves: continue
            if is_torch_tensor(v) or is_numpy_array(v):
                if len(v.shape) > self.batch_dim:
                    batch_size = v.shape[self.batch_dim]
                    break
        return batch_size



    def to(self, device):
        """Move all data to a pytorch Device.

        This does NOT modify the original GraphInput object but returns a new
        one. """
        # assert all(isinstance(t, torch.Tensor) for _, t in self._values.items())
        new_values = {k: v.to(device) if is_torch_tensor(v) else v for k, v in self._values.items()}
        return GraphInput(
            new_values,
            cache_results=self.cache_results,
            batched=self.batched,
            batch_dim=self.batch_dim, keys=self.keys, key_leaves=self.key_leaves)
