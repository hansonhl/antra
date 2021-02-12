from compgraph import GraphNode, GraphInput, Intervention, Location
import itertools
import torch

from typing import Optional
from collections import deque

# TODO: add type hints

class ComputationGraph:
    def __init__(self, root: GraphNode,
                 root_output_device: Optional[torch.device]=None):
        """
        Constructs a computation graph by traversing from a root
        :param root: Root node
        :param output_device: (For models that run on pytorch) transfer final
            root output to a given device
        """
        self.root = root
        self.root_output_device = root_output_device
        self.nodes = {}
        self.results_cache = {}  # caches final results compute(), intervene()
        self.leaves = set()
        self.validate_graph()
        self.cache_device = None

    def get_from_cache(self, inputs):
        if inputs.batched: return None

        result = self.results_cache.get(inputs, None)
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            output_device = self.result_output_device_dict[inputs]
            if output_device != self.cache_device:
                return result.to(output_device)
        return result

    def save_to_cache(self, inputs, result):
        if not inputs.cache_results or inputs.batched:
            return

        result_for_cache = result
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            if result.device != self.cache_device:
                result_for_cache = result.to(self.cache_device)
            self.result_output_device_dict[inputs] = result.device

        self.results_cache[inputs] = result_for_cache

    def set_cache_device(self, cache_device):
        self.cache_device = cache_device
        self.result_output_device_dict = {}

        for node in self.nodes.values():
            node.cache_device = cache_device

    def get_nodes_and_dependencies(self):
        nodes = [node_name for node_name in self.nodes]
        dependencies = {self.root.name: set()}
        def fill_dependencies(node):
            for child in node.children:
                if child in dependencies:
                    dependencies[child.name].add(node.name)
                else:
                    dependencies[child.name] = {node.name}
                fill_dependencies(child)
        fill_dependencies(self.root)
        return nodes, dependencies

    def get_indices(self,node):
        length = None
        for key in self.nodes[node].base_cache:
            length = max(self.nodes[node].base_cache[key].shape)
        indices = []
        for i in range(length):
            for subset in itertools.combinations({x for x in range(0, length)},i+1):
                subset = list(subset)
                subset.sort()
                indices.append(Location()[subset])
        return indices

    def get_locations(self, root_locations, unwanted_low_nodes=None):
        root_nodes = []
        for location in root_locations:
            for node_name in location:
                root_nodes.append(self.nodes[node_name])
        viable_nodes = None
        for root_node in root_nodes:
            current_nodes = set()
            def descendants(node):
                for child in node.children:
                    current_nodes.add(child.name)
                    descendants(child)
            descendants(root_node)
            if viable_nodes is None:
                viable_nodes = current_nodes
            else:
                viable_nodes = viable_nodes.intersection(current_nodes)
        result = []
        for viable_node in viable_nodes:
            if unwanted_low_nodes and viable_node in unwanted_low_nodes:
                continue
            for index in self.get_indices(viable_node):
                result.append({viable_node:index})
        return result

    def validate_graph(self):
        """Validate the structure of the computational graph

        :raise: `RuntimeError` if something goes wrong
        """

        # TODO: check for cycles
        def add_node(node):
            if node.name in self.nodes:
                if self.nodes[node.name] is not node:
                    raise RuntimeError(
                        "Two different nodes cannot have the same name!")
                else:
                    return
            self.nodes[node.name] = node
            if len(node.children) == 0:
                self.leaves.add(node)
            for child in node.children:
                add_node(child)
        add_node(self.root)

    def validate_inputs(self, inputs):
        """
        Check if an input is provided for each leaf node
        :raise: `RuntimeError` if something goes wrong
        """
        for node in self.leaves:
            if node.name not in inputs:
                raise RuntimeError(
                    "input value not provided for leaf node %s" % node.name)

    def validate_interv(self, intervention):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param intervention:  intervention experiment in question
        :raise: `RuntimeError` if something goes wrong
        """
        self.validate_inputs(intervention.base)

        if intervention.intervention is None:
            raise RuntimeError("Must specify some kind of intervention!")

        for name in intervention.intervention.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                   "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs, store_cache=True, iterative=False):
        """
        Run forward pass through graph with a given set of inputs

        :param inputs:
        :param store_cache:
        :return:
        """
        result = self.get_from_cache(inputs)
        if not result:
            self.validate_inputs(inputs)
            if iterative:
                result = self._iterative_compute(inputs)
            else:
                result = self.root.compute(inputs)

        if store_cache:
            self.save_to_cache(inputs, result)

        if isinstance(result, torch.Tensor) and self.root_output_device:
            result = result.to(self.root_output_device)
        return result

    def _iterative_compute(self, inputs):
        """
        stack = deque()
        stack.append(self.root)

        while len(stack) > 0:
            curr_node = stack[-1]
            for c in curr_node.children:
                stack.append(c)
        """
        raise NotImplementedError

    def intervene(self, intervention, store_cache=True):
        """
        Run intervention on computation graph.

        :param intervention:
        :param store_cache:
        :return:
        """
        base_res = self.compute(intervention.base)

        interv_res = self.get_from_cache(intervention)
        self.validate_interv(intervention)
        intervention.find_affected_nodes(self)
        if not interv_res:
            interv_res = self.root.compute(intervention)
            if store_cache:
                self.save_to_cache(intervention, interv_res)

        return base_res, interv_res

    def clear_caches(self):
        def clear_cache(node):
            node.clear_caches()
            for c in node.children:
                clear_cache(c)

        clear_cache(self.root)
        del self.results_cache
        self.results_cache = {}

        if hasattr(self, "result_output_device_dict"):
            del self.result_output_device_dict
            self.result_output_device_dict = {}

    def get_result(self, node_name, x):
        node = self.nodes[node_name]
        res = None

        if isinstance(x, GraphInput):
            res = node.compute(x)
            # if x not in node.base_cache:
            #     self.compute(x)
            # res = node.base_cache[x]
        elif isinstance(x, Intervention):
            x.find_affected_nodes(self)
            if x.base not in node.base_cache or x not in node.interv_cache:
                base_res = self.compute(x.base)
                self.root.compute(x)

            res = node.compute(x)
            # if node.name not in x.affected_nodes:
            #     res = node.base_cache[x.base]
            # else:
            #     res = node.interv_cache[x]
        else:
            raise RuntimeError(
                "get_result requires a GraphInput or Intervention "
                "object!")

        if isinstance(res, torch.Tensor) and self.root_output_device:
            res = res.to(self.root_output_device)
        return res

    def get_state_dict(self):
        return {
            "base_caches": {
                node_name: node.base_cache for node_name, node in self.nodes.items() if node.cache_results
            },
            "interv_caches": {
                node_name: node.interv_cache for node_name, node in self.nodes.items() if node.cache_results
            },
            "base_output_devices": {
                node_name: node.base_output_devices for node_name, node in self.nodes.items() if node.cache_results
            },
            "interv_output_devices": {
                node_name: node.interv_output_devices for node_name, node in self.nodes.items() if node.cache_results
            }
        }

    def set_state_dict(self, d):
        pass