from .graph import ComputationGraph
from .graph_node import GraphNode

from typing import Any, Dict, List, Callable, Set


def find_topological_order(parent2children: Dict[str, List[str]], root):
    ordering = []
    visited = set()
    def recursive_call(node):
        visited.add(node)
        for i in range(len(parent2children[node])-1, -1, -1):
            if parent2children[node][i] not in visited:
                recursive_call(parent2children[node][i])
        ordering.append(node)
    recursive_call(root)
    return ordering[::-1]


class AbstractableCompGraph(ComputationGraph):
    def __init__(self, graph: ComputationGraph, abstract_nodes: List[str]):
        """ An abstractable compgraph structure.

        :param graph:
        :param abstract_nodes: Names of intermediate nodes that we would like
            to keep, while abstracting away all other nodes
        """
        self._parent2children = {
            node_name: [c.name for c in node.children]
            for node_name, node in graph.nodes.items()
        }
        self._leaf_nodes = {k for k, v in self._parent2children.items() if len(v) == 0}
        self._root_node_name = graph.root.name
        self._forward_functions = {
            node_name: node.forward for node_name, node in graph.nodes.items()
        }

        self._topological_order = find_topological_order(self._parent2children,
                                                         self._root_node_name)

        self._validate_full_graph()

        root = self._generate_abstract_graph(abstract_nodes)
        super(AbstractableCompGraph, self).__init__(root)

    def _validate_full_graph(self):
        # all nodes must be present in keys of full_graph dict
        nodes = set(n for _, nodes in self._parent2children.items() for n in nodes)
        keys = set(self._parent2children.keys())
        diff = nodes.difference(keys)
        if len(diff) > 0:
            raise RuntimeError(f"All nodes in the underlying graph should be "
                               f"in the keys of the dict. Missing: {diff}")

        # each non-leaf node should have a forward function
        no_forward = nodes.difference(self._leaf_nodes)\
            .difference(set(self._forward_functions.keys()))
        if len(no_forward) > 0:
            raise RuntimeError(f"These nodes are missing an associated forward "
                               f"function: {no_forward}")

        # TODO: check topological order
        # TODO: check for cycles

    def _generate_abstract_graph(self, abstract_nodes: List[str]) -> GraphNode:
        # rearrange nodes in reverse topological order
        relevant_nodes = self._get_node_names(abstract_nodes)
        relevant_node_set = set(relevant_nodes)

        # define input leaf node
        node_dict = {name: GraphNode.leaf(name) for name in self._leaf_nodes}

        for node_name in relevant_nodes:
            if node_name in self._leaf_nodes:
                continue

            curr_children = self._get_children(node_name, relevant_node_set)
            args = [node_dict[child] for child in curr_children]
            forward = self._generate_forward_function(node_name, curr_children)
            node_dict[node_name] = GraphNode(*args, name=node_name,
                                               forward=forward)

        return node_dict[self._root_node_name]

    def _get_node_names(self, abstract_nodes: List[str]) -> List[str]:
        """ Get topologically ordered list of node names in final compgraph,
        given intermediate nodes"""
        abstract_nodes = set(abstract_nodes)
        if self._root_node_name not in abstract_nodes:
            abstract_nodes.add(self._root_node_name)

        for input_node_name in self._leaf_nodes:
            if input_node_name not in abstract_nodes:
                abstract_nodes.add(input_node_name)

        res = []
        for i in range(len(self._topological_order) - 1, -1, -1):
            if self._topological_order[i] in abstract_nodes:
                res.append(self._topological_order[i])
        return res

    def _get_children(self, abstract_node: str, abstract_node_set: Set[str]) \
            -> List[str]:
        """ Get immediate children in abstracted graph given an abstract node """
        res = []
        stack = [abstract_node]
        visited = set()

        while len(stack) > 0:
            curr_node = stack.pop()
            visited.add(curr_node)
            if curr_node != abstract_node and curr_node in abstract_node_set:
                res.append(curr_node)
            else:
                for i in range(len(self._parent2children[curr_node]) - 1, -1, -1):
                    child = self._parent2children[curr_node][i]
                    if child not in visited:
                        stack.append(child)
        return res

    def _generate_forward_function(self, abstracted_node: str,
                                   children: List[str]) -> Callable:
        """ Generate forward function to construct an abstract node """
        child_name_to_idx = {name: i for i, name in enumerate(children)}

        def _forward(*args):
            if len(args) != len(children):
                raise ValueError(f"Got {len(args)} arguments to forward fxn of "
                                 f"{abstracted_node}, expected {len(children)}")

            def _implicit_call(node_name: str) -> Any:
                if node_name in child_name_to_idx:
                    child_idx_in_args = child_name_to_idx[node_name]
                    return args[child_idx_in_args]
                else:
                    res = [_implicit_call(child) for child in self._parent2children[node_name]]
                    current_f = self._forward_functions[node_name]
                    return current_f(*res)

            return _implicit_call(abstracted_node)

        return _forward
