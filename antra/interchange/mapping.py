import copy
import itertools
from antra import *


from typing import *

NodeName = str
AbstractionMappingType = Dict[NodeName, Dict[NodeName, LocationType]]

def get_nodes_and_dependencies(graph: ComputationGraph):
        nodes = [node_name for node_name in graph.nodes]
        dependencies = {graph.root.name: set()}
        def fill_dependencies(node):
            for child in node.children:
                if child in dependencies:
                    dependencies[child.name].add(node.name)
                else:
                    dependencies[child.name] = {node.name}
                fill_dependencies(child)
        fill_dependencies(graph.root)
        return nodes, dependencies

def get_indices(graph: ComputationGraph, node: NodeName):
    length = None
    for key in graph.nodes[node].base_cache:
        length = max(graph.nodes[node].base_cache[key].shape)
    indices = []
    for i in range(length):
        for subset in itertools.combinations({x for x in range(0, length)},i+1):
            subset = list(subset)
            subset.sort()
            indices.append(Location()[subset])
    return indices

def get_locations(graph: ComputationGraph, root_locations: Sequence[NodeName],
                  unwanted_low_nodes: Optional[List[NodeName]]=None):
    root_nodes = []
    for location in root_locations:
        for node_name in location:
            root_nodes.append(graph.nodes[node_name])
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
        for index in get_indices(graph, viable_node):
            result.append({viable_node:index})
    return result

def create_possible_mappings(
        low_model: ComputationGraph,
        high_model: ComputationGraph,
        fixed_assignments: AbstractionMappingType,
        unwanted_low_nodes: Optional[List[NodeName]]=None) \
        -> List[AbstractionMappingType]:
    """
    :param low_model:
    :param high_model:
    :param fixed_assignments: dict: str name of highlevel node -> (dict: name of low level node -> locations)
    :return: list(dict: str name of highlevel node -> (dict: name of low level node -> locations))
    """
    class MappingCertificate:
        def __init__(self, partial_mapping, high_nodes, dependencies):
            self.partial_mapping = partial_mapping
            self.high_nodes = [x  for x in high_nodes if x not in fixed_assignments]
            self.dependencies = dependencies
            for high_node in fixed_assignments:
                partial_mapping[high_node] = fixed_assignments[high_node]

        def remove_high_node(self):
            self.high_nodes = self.high_nodes[1:]

        def compatible_splits(self,split1, split2):
            return False

        def compatible_location(self, location):
            for high_node in self.partial_mapping:
                for low_node in self.partial_mapping[high_node]:
                    if low_node in location:
                        if not self.compatible_splits(self.partial_mapping[high_node][low_node], location[low_node]):
                            return False
            return True

        def set_assignment_list(self):
            self.assignment_list = []
            if len(self.high_nodes) == 0:
                return
            #grab the next high-level node
            high_node = self.high_nodes[0]
            dependent_high_nodes = self.dependencies[high_node]
            #cycle through the potential locations in the low-level model we can map the high-level node to
            locations = get_locations(low_model, [self.partial_mapping[x] for x in dependent_high_nodes], unwanted_low_nodes)
            for location in locations:
                    if self.compatible_location(location):
                        self.assignment_list.append((high_node, location))

        def add_assignment(self):
            #add a new assignment to the partially constructed mapping
            self.remove_high_node()
            high_node,low_location = self.assignment_list[0]
            self.partial_mapping[high_node] = low_location
            if len(self.high_nodes) != 0:
                self.set_assignment_list()

        def next_assignment(self):
            #move on to the next assignment
            self.assignment_list = self.assignment_list[1:]

    mappings = []

    def accept(certificate):
        if len(certificate.high_nodes) == 0:
            return True
        return False

    def next(certificate):
        if len(certificate.assignment_list) == 0:
            return None
        new_certificate = copy.deepcopy(certificate)
        #Add in a assignmen to the mapping
        new_certificate.add_assignment()
        #Cycle the original map to the next assignment that could have been added
        certificate.next_assignment()
        #return the partial map
        return new_certificate

    def root():
        high_nodes, dependencies = get_nodes_and_dependencies(high_model)
        certificate = MappingCertificate(dict(), high_nodes, dependencies)
        certificate.set_assignment_list()
        return certificate

    def backtrack(certificate):
        if accept(certificate):
            mappings.append(certificate.partial_mapping)
            return
        next_certificate = next(certificate)
        while next_certificate is not None:
            backtrack(next_certificate)
            next_certificate = next(certificate)
    backtrack(root())
    return mappings