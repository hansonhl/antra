import sys

from .intervention import Intervention
from .graph_input import GraphInput
from .utils import copy_helper

from typing import Union, Callable, Dict

if "torch" in sys.modules:
    import compgraph.torch_utils as torch_utils

# TODO: add type hints

class GraphNode:
    def __init__(self, *args, name: str=None, forward: Callable=None, cache_results: bool=True):
        """Construct a computation graph node, can be used as function decorator

        This constructor is invoked when `@GraphNode()` decorates a function.
        When used as a decorator, the `*args` become parameters of the decorator

        :param args: GraphNode objects that are the children of this node
        :param name: the name of the node. If not given, this will be the name
            of the function that it decorates by default
        :param forward: the name of the forward function
        :param cache_results: whether this node caches results. Cannot perform
            interventions
        """
        self.children = args
        self.cache_results = cache_results
        if self.cache_results:
            self.base_cache = {}  # stores results of non-intervened runs
            self.interv_cache = {}  # stores results of intervened experiments

            # keep track which devices the results are originally stored in
            self.base_output_devices = {}
            self.interv_output_devices = {}


        self.name = name
        if forward:
            self.forward = forward
            if name is None:
                self.name = forward.__name__

        # Saving the results in their original devices by default
        self.cache_device = None

    def __call__(self, f):
        """Invoked immediately after `__init__` during `@GraphNode()` decoration

        :param f: the function to which the decorator is attached
        :return: a new GraphNode object
        """
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        # adding the decorator GraphNode on a function returns GraphNode object
        return self

    def __repr__(self):
        return "GraphNode(\"%s\")" % self.name

    @classmethod
    def leaf(cls, name: str):
        """Construct a leaf node with a given name."""
        return cls(name=name, forward=lambda x: x, cache_results=False)

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, values):
        self._children = list(values)
        self._children_dict = {c.name: c for c in self._children}

    @property
    def children_dict(self):
        if hasattr(self, "_children_dict"):
            return self._children_dict
        else:
            self._children_dict = {c.name: c for c in self._children}
            return self._children_dict

    def _get_from_cache(self, inputs: Union[GraphInput, Intervention],
                        from_interv: bool):
        if not self.cache_results: return None

        cache = self.interv_cache if from_interv else self.base_cache
        output_device_dict = self.interv_output_devices if from_interv \
            else self.base_output_devices
        assert from_interv and isinstance(inputs, Intervention) or \
               (not from_interv) and isinstance(inputs, GraphInput)

        if inputs.batched:
            result = torch_utils.get_batch_from_cache(
                inputs, cache, self.cache_device, output_device_dict)
        else:
            result = cache.get(inputs.keys, None)

            if self.cache_device is not None:
                output_device = output_device_dict[inputs.keys]
                if output_device != self.cache_device:
                    return result.to(output_device)
        return result

    def _save_to_cache(self, inputs, result, to_interv):
        if not self.cache_results or not inputs.cache_results:
            return

        cache = self.interv_cache if to_interv else self.base_cache
        output_device_dict = self.interv_output_devices if to_interv \
            else self.base_output_devices

        if inputs.batched:
            torch_utils.save_batch_to_cache(inputs, result, cache, self.cache_device, output_device_dict)
        else:
            result_for_cache = result
            if self.cache_device is not None:
                if result.device != self.cache_device:
                    result_for_cache = result.to(self.cache_device)
                output_device_dict[inputs.keys] = result.device
            cache[inputs.keys] = result_for_cache

    def compute(self, inputs: Union[GraphInput, Intervention]):
        """Compute the output of a node

        :param inputs: Can be a GraphInput object or an Intervention object
        :return: result of forward function
        """
        # check if intervention is happening in this run
        intervention = None
        is_affected = False
        if isinstance(inputs, Intervention):
            intervention = inputs
            inputs = intervention.base
            if intervention.affected_nodes is None:
                raise RuntimeError("Must find affected nodes with respect to "
                                   "a graph before intervening")
            is_affected = self.name in intervention.affected_nodes

        result = self._get_from_cache(intervention if is_affected else inputs,
                                      is_affected)

        if result is not None:
            return result
        else:
            if intervention and self.name in intervention.intervention:
                if self.name in intervention.location:
                    # intervene a specific location in a vector/tensor
                    # result = self.base_cache.get(inputs, None)
                    if not self.cache_results:
                        raise RuntimeError(f"Cannot intervene on node {self.name} "
                                           f"because its results are not cached")
                    result = self._get_from_cache(inputs, from_interv=False)
                    if result is None:
                        raise RuntimeError(
                            f"Must compute result of node \"{self.name}\" without intervention once "
                            "before intervening (base: %s, intervention: %s)"
                            % (intervention.base, intervention.intervention))
                    result = copy_helper(result)
                    idx = intervention.location[self.name]
                    result[idx] = intervention.intervention[self.name]
                else:
                    # replace the whole tensor
                    result = intervention.intervention[self.name]
                if len(self.children) != 0:
                    for child in self.children:
                        child_res = child.compute(
                            inputs if intervention is None else intervention)
            else:
                if len(self.children) == 0:
                    # leaf
                    values = inputs[self.name]
                    # if isinstance(values, list) or isinstance(values, tuple):
                    #     result = self.forward(*values)
                    # else:
                    result = self.forward(values)
                else:
                    # non-leaf node
                    children_res = []
                    for child in self.children:
                        child_res = child.compute(
                            inputs if intervention is None else intervention)
                        children_res.append(child_res)
                    result = self.forward(*children_res)

            self._save_to_cache(intervention if is_affected else inputs,
                                result, is_affected)

            return result

    def clear_caches(self):
        """Clear all caches"""
        if hasattr(self, "base_cache"):
            del self.base_cache
            self.base_cache = {}
        if hasattr(self, "interv_cache"):
            del self.interv_cache
            self.interv_cache = {}
        if hasattr(self, "base_output_devices"):
            del self.base_output_devices
            self.base_output_devices = {}
        if hasattr(self, "interv_output_devices"):
            del self.interv_output_devices
            self.interv_output_devices = {}