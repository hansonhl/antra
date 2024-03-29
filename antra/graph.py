import itertools

from .intervention import Intervention
from .graph_node import GraphNode
from .graph_input import GraphInput
from .utils import serialize

from typing import *
from collections import defaultdict

import logging

logging.getLogger(__name__)


class ComputationGraph:
    def __init__(self, root: GraphNode):
        """
        Constructs a computation graph by traversing from a root
        :param root: Root node
        :param output_device: (For models that run on pytorch) transfer final
            root output to a given device
        """
        self.root: GraphNode = root
        self.nodes: Dict[str, GraphNode] = {}
        self.leaves: Set[GraphNode] = self._validate_graph()

    def _validate_graph(self):
        """ Validate the structure of the computational graph

        :raise: `RuntimeError` if something goes wrong
        :return: set of leaf nodes
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

        for name in intervention.intervention._values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                   "in computation graph: %s" % name)

    def compute(self, inputs: GraphInput):
        """ Run forward pass through graph with a given set of inputs

        :param inputs:
        :return:
        """
        return self.root.compute(inputs)

    def compute_all_nodes(self, inputs: GraphInput) -> Dict[str, Any]:
        """ Get the output for each node in the graph. This will temporarily
        force the graph to cache the results of the inputs.

        :param inputs:
        :return:
        """
        res_dict = {}
        self.root.compute(inputs, res_dict=res_dict)
        return res_dict

        # no_caching = not inputs.cache_results
        # if no_caching:
        #     inputs.cache_results = True
        #
        # res_dict = {self.root.name: self.root.compute(inputs)}
        #
        # for node_name, node in self.nodes.items():
        #     if node_name != self.root.name:
        #         res_dict[node_name] = node.compute(inputs)
        #
        # if no_caching:
        #     self.clear_caches(inputs)
        #     inputs.cache_results = False
        #
        # return res_dict

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

    def intervene_all_nodes(self, intervention: Intervention) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Get the output for every node after an intervention in the graph.

        :param inputs:
        :return:
        """
        base_res_dict = self.compute_all_nodes(intervention.base)

        self._validate_interv(intervention)
        intervention.find_affected_nodes(self)

        ivn_res_dict = {}
        self.root.compute(intervention, ivn_res_dict)
        return base_res_dict, ivn_res_dict

        # no_caching = not intervention.cache_results
        # if no_caching:
        #     intervention.cache_results = True
        #
        # res_dict = {self.root.name: self.root.compute(intervention)}
        #
        # for node_name, node in self.nodes.items():
        #     if node_name != self.root.name:
        #         res_dict[node_name] = node.compute(intervention)
        #
        # if no_caching:
        #     self.clear_caches(intervention)
        #     intervention.cache_results = False

    def set_cache_device(self, cache_device):
        for node in self.nodes.values():
            node.cache_device = cache_device

    def clear_caches(self,
                     inputs: Union[GraphInput, Intervention, None] = None,
                     force=False):
        """ Clear all caches, or clear a specified cache entry for a given input.

        :param inputs:
        :return:
        """
        for node in self.nodes.values():
            if force:
                node.clear_caches(inputs)
            elif node.cache_results:
                node.clear_caches(inputs)

    def compute_node(self, node_name: str, x: GraphInput):
        """ Compute the value of a node in the graph without any interventions

        :param node_name: name of node
        :param x: GraphInput object
        :return: output from node
        """
        try:
            return self.nodes[node_name].compute(x)
        except KeyError as e:
            logging.error(f'Invalid key {node_name}. Valid keys are {self.nodes.keys()}')
            raise e

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
        
    def compute_node_partitions(self, input1, input2, ignore_nodes=None):
        partition = defaultdict(set)
        if ignore_nodes is None:
            ignore_nodes = set()
        for node in self.nodes:
            if node in ignore_nodes: continue
            ivn = Intervention(
                input1, {node: self.compute_node(node,input2)},
                cache_results=False
            )
            _, ivn_res = self.intervene(ivn)
            ser_ivn_res = serialize(ivn_res)
            partition[ser_ivn_res].add(node)
        return partition

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
