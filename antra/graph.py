import itertools

from .intervention import Intervention
from .graph_node import GraphNode
from .graph_input import GraphInput
from .location import Location

from typing import Dict, Any, Union
from collections import deque

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
        self.leaves = self._validate_graph()

    def _validate_graph(self):
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

    def _validate_interv(self, intervention: Intervention):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param intervention:  intervention experiment in question
        :raise: `RuntimeError` if something goes wrong
        """
        # if not intervention.intervention:
        #     raise RuntimeError("Must specify some kind of intervention!")

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

        self._validate_interv(intervention)
        intervention.find_affected_nodes(self)

        interv_res = self.root.compute(intervention)

        return base_res, interv_res

    def clear_caches(self):
        """ Clear all caches.
        :return: None
        """
        def clear_cache(node):
            node.clear_caches()
            for c in node.children:
                clear_cache(c)

        clear_cache(self.root)

    def compute_node(self, node_name: str, x: GraphInput):
        """ Compute the value of a node in the graph without any interventions

        :param node_name: name of node
        :param x: GraphInput object
        :return: output from node
        """
        return self.nodes[node_name].compute(x)

    def intervene_node(self, node_name: str, x: Intervention):
        """ Compute the value of a node during an intervention

        :param node_name: name of node
        :param x: Intervention object
        :return: output from node
        """
        node = self.nodes[node_name]
        base_res = node.compute(x.base)
        self._validate_interv(x)
        x.find_affected_nodes(self)

        interv_res = node.compute(x)
        return base_res, interv_res

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