import itertools

from .intervention import Intervention
from .graph_node import GraphNode
from .graph_input import GraphInput
from .location import Location

from typing import Dict, Any, Union
from collections import deque

# TODO: add type hints
# TODO: index of input keys to reduce cache space

class ComputationGraph:
    def __init__(self, root: GraphNode):
        """
        Constructs a computation graph by traversing from a root
        :param root: Root node
        :param output_device: (For models that run on pytorch) transfer final
            root output to a given device
        """
        self.root = root
        self.nodes = {}
        self.leaves = self.validate_graph()

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
        leaves = set()
        def add_node(node):
            if node.name in self.nodes:
                if self.nodes[node.name] is not node:
                    raise RuntimeError(
                        "Two different nodes cannot have the same name!")
                else:
                    return
            self.nodes[node.name] = node
            if len(node.children) == 0:
                leaves.add(node)
            for child in node.children:
                add_node(child)
        add_node(self.root)
        return leaves

    def validate_inputs(self, inputs: GraphInput):
        """
        Check if an input is provided for each leaf node
        :raise: `RuntimeError` if something goes wrong
        """
        for node in self.leaves:
            if node.name not in inputs:
                raise RuntimeError(
                    "input value not provided for leaf node %s" % node.name)

    def validate_interv(self, intervention: Intervention):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param intervention:  intervention experiment in question
        :raise: `RuntimeError` if something goes wrong
        """
        self.validate_inputs(intervention.base)

        if not intervention.intervention:
            raise RuntimeError("Must specify some kind of intervention!")

        for name in intervention.intervention.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                   "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs: GraphInput):
        """
        Run forward pass through graph with a given set of inputs

        :param inputs:
        :return:
        """
        self.validate_inputs(inputs)
        return self.root.compute(inputs)

    def _iterative_compute(self, inputs: GraphInput):
        """
        stack = deque()
        stack.append(self.root)

        while len(stack) > 0:
            curr_node = stack[-1]
            for c in curr_node.children:
                stack.append(c)
        """
        raise NotImplementedError

    def intervene(self, intervention: Intervention):
        """ Run intervention on computation graph.

        :param intervention: Intervention object.
        :return: base result and intervention result
        """
        base_res = self.compute(intervention.base)

        self.validate_interv(intervention)
        intervention.find_affected_nodes(self)

        interv_res = self.root.compute(intervention)

        return base_res, interv_res

    def clear_caches(self):
        def clear_cache(node):
            node.clear_caches()
            for c in node.children:
                clear_cache(c)

        clear_cache(self.root)

    def compute_node(self, node_name: str, x: Union[GraphInput, Intervention]):
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
            raise RuntimeError("compute_node requires a GraphInput or Intervention object!")

        # if self.root_output_device:
        #     res = res.to(self.root_output_device)
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