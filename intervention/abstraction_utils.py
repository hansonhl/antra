# This script is the same as `causal_abstraction/abstraction_utils.py`.
# This script is depended on by abstraction_batched.py
# It is duplicated here to support old pickled experiment result files.

import copy
import numpy as np
import torch
from intervention.graph_input import GraphInput
from intervention.intervention import Intervention
from intervention.location import Location

from intervention.utils import serialize, deserialize


def create_possible_mappings(low_model, high_model, fixed_assignments=dict(),
                             unwanted_low_nodes=None):
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
            for location in low_model.get_locations([self.partial_mapping[x] for x in dependent_high_nodes], unwanted_low_nodes):
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
        high_nodes, dependencies = high_model.get_nodes_and_dependencies()
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


def get_value(high_model, high_node, high_intervention):
    return high_model.get_result(high_node, high_intervention)

def create_new_realizations(low_model, high_model, high_node, mapping, low_intervention, high_intervention):
    """H: Return a direct mapping between each high intermediate node value and low intermediate node value,
    given a pair of interventions"""
    new_realizations = dict()
    new_realizations_to_inputs = dict()
    def fill_new_realizations(high_node, mapping, low_intervention, high_intervention):
        high_value = get_value(high_model, high_node, high_intervention)
        low_value = None
        for low_node in mapping[high_node]:
            low_value = low_model.get_result(low_node,low_intervention)[mapping[high_node][low_node]] # index into location
        for child in high_model.nodes[high_node].children:
            fill_new_realizations(child.name, mapping, low_intervention, high_intervention)
        if high_node in high_intervention.intervention.values or high_model.nodes[high_node] in high_model.leaves or high_node == high_model.root.name:
            return # H: ignore intervened nodes, leaf nodes and root nodes

        # H: in first dry run without interv, low_value and high_value will be intermediate nodes
        # H: make hypothetical mapping between <high node and value> and <low node and value>
        low_string = serialize(low_value)
        high_string = serialize(high_value)
        new_realizations[(high_node, high_string)] = low_string
        new_realizations_to_inputs[(low_string, high_node)] = low_intervention
        # H: The low intervention that potentially realizes this high-level intermediate value

    fill_new_realizations(high_node, mapping, low_intervention, high_intervention)
    return new_realizations, new_realizations_to_inputs


def get_potential_realizations(new_realizations, total_realizations, high_node, high_model, new_high_intervention):
    partial_realizations = [dict()]
    for high_node2 in new_high_intervention.intervention.values:
        high_value2 = serialize(get_value(high_model, high_node2, new_high_intervention))
        if high_model.nodes[high_node2] in high_model.leaves or high_node2 == high_model.root.name:
            continue # H: ignore leaves and root
        if high_node2 == high_node:
            # H: this is definitely the case for now
            high_value = serialize(get_value(high_model, high_node, new_high_intervention))
            new_partial_realizations = []
            for partial_realization in partial_realizations:
                if (high_node, high_value) not in new_realizations:
                    return []
                partial_realization[(high_node, high_value)] = new_realizations[(high_node, high_value)]
                # H: fetch low level node value that is hypothesized to correspond to intervention value
                new_partial_realizations.append(partial_realization)
            partial_realizations = new_partial_realizations
        else:
            new_partial_realizations = []
            for partial_realization in partial_realizations:
                if (high_node2,high_value2) in new_realizations:
                    partial_realization_copy = copy.deepcopy(partial_realization)
                    partial_realization_copy[(high_node2, high_value2)] = new_realizations[(high_node2, high_value2)]
                    new_partial_realizations.append(partial_realization_copy)
                else:
                    if (high_node2, high_value2) not in total_realizations:
                        return []
                    for low_value in total_realizations[(high_node2, high_value2)]:
                        partial_realization_copy = copy.deepcopy(partial_realization)
                        partial_realization_copy[(high_node2, high_value2)] = low_value
                        new_partial_realizations.append(partial_realization_copy)
            partial_realizations = new_partial_realizations
    return partial_realizations


def high_to_low(high_model, high_intervention,realization, mapping, input_mapping):
    intervention = dict()
    location = dict()
    base = dict()
    for high_node in high_model.leaves:
        for low_node in mapping[high_node.name]:
            base[low_node] = input_mapping(get_value(high_model, high_node.name, high_intervention))
    for high_node in high_intervention.intervention.values:
        high_value = serialize(get_value(high_model, high_node, high_intervention))
        for low_node in mapping[high_node]:
            low_value = deserialize(realization[(high_node, high_value)])
            intervention[low_node] = low_value
            location[low_node] = mapping[high_node][low_node]
    return Intervention(base,intervention,location)

def truncate(x):
    return x

def test_mapping(low_model,high_model,high_inputs,total_high_interventions,mapping, input_mapping):
    low_and_high_interventions = [(high_to_low(high_model, high_intervention, dict(), mapping, input_mapping), high_intervention) for high_intervention in high_inputs]
    total_realizations = dict()
    result = dict()
    realizations_to_inputs = dict()
    counter =0
    while len(low_and_high_interventions) !=0:
        #print(mapping)
        #print(low_and_high_interventions[0][0].intervention.values,low_and_high_interventions[0][1].intervention.values)
        #print("total len:", len(total_realizations))
        #print("\n", "totalreals:")
        if counter %5000 == 0 :
            print("low high len:",len(low_and_high_interventions))
            for key in total_realizations:
                print(key, len(total_realizations[key]))
                #for realization in total_realizations[key]:
                    #print(np.fromstring(realization))
        #print("\n\n")
        # H: pop one low, high intervention pair
        curr_low_intervention,curr_high_intervention = low_and_high_interventions[0]
        low_and_high_interventions = low_and_high_interventions[1:]
        # store whether the output matches
        high_output = get_value(high_model, high_model.root.name, curr_high_intervention)
        for low_node in mapping[high_model.root.name]:
            low_output = low_model.get_result(low_node,curr_low_intervention)
        result[(curr_low_intervention, curr_high_intervention)] = low_output == high_output
        # update the realizations H: for this low, high intervention pair
        new_realizations, new_realizations_to_inputs = create_new_realizations(low_model, high_model,high_model.root.name, mapping, curr_low_intervention, curr_high_intervention)
        # add on the new interventions that need to be checked
        used_high_interventions = set()
        for new_high_intervention in total_high_interventions:
            for high_node, high_value in new_realizations: # H: just one pair for now
                if high_node in new_high_intervention.intervention.values and new_high_intervention not in used_high_interventions:
                    # H: pick an intervention from total_high_interventions (there is only one intermediate high node so it's okay)
                    realizations = get_potential_realizations(new_realizations, total_realizations, high_node, high_model, new_high_intervention)
                    # H: realizations: (high_node, high_value) -> stringified_low_value
                    # H: realizations may be empty , because
                    for realization in realizations:
                        if (high_node, truncate(high_value)) in total_realizations:
                            if new_realizations[(high_node, high_value)] in total_realizations[(high_node, truncate(high_value))]:
                                continue

                        # H: based on realizations, create new low intervention such that
                        # intervention value corresponds to that in stringified_low_value
                        new_low_intervention = high_to_low(high_model,new_high_intervention, realization,mapping, input_mapping)

                        low_and_high_interventions.append((new_low_intervention, new_high_intervention))
                        used_high_interventions.add(new_high_intervention)
        #merge the new_realizations into the total realizations
        for high_node,high_value in new_realizations:
            if (high_node, truncate(high_value)) in total_realizations:
                total_realizations[(high_node, truncate(high_value))].add(new_realizations[(high_node, high_value)])
            else:
                total_realizations[(high_node, truncate(high_value))] = {new_realizations[(high_node, high_value)]}
            #total_realizations[(high_node, high_value)] = list(set(total_realizations[(high_node, high_value)]))
        for realization in new_realizations_to_inputs:
            realizations_to_inputs[realization] = new_realizations_to_inputs[realization]
        counter +=1
        if counter > 100 and False:
            raise RuntimeError
    #print(mapping)
    #for key in total_realizations:
        #print(key, len(total_realizations[key]))
        #for realization in total_realizations[key]:
        #    print(np.fromstring(realization))
    return (result,realizations_to_inputs)



def find_abstractions(low_model, high_model, high_inputs, total_high_interventions, fixed_assignments, input_mapping, unwanted_low_nodes=None):
    """

    :param low_model: CompGraph
    :param high_model: CompGraph
    :param high_inputs: A list of Intervention objects that only have a base, but no intervention. These are the inputs you want to cover.
    :param total_high_interventions: A list of Intervention objects that have a base and an intervention.
    :param fixed_assignments:This is a dictionary mappping from high level nodes to a dictionary mapping from low level nodes to Location objects.
        Typically, this will look something like: {x:{x:Location()[:]} for x in ["root", "leaf1",  "leaf2", "leaf3"]}
        Only input leaves and roots.
    :param input_mapping: A function that maps high level leaf inputs to low level leaf inputs
    :return:
        list(
            tuple(
                tuple (
                    dict: "results" tuple(low interv, high interv) -> bool (True if two interventions result in same output),
                    dict: "realizations_to_inputs" (serialized value of intervention low level run, string node name)  ->  low interv object
                )
                dict: "mapping" str name of highlevel node -> (dict: name of low level node -> locations)
            )
        )
    """
    result = []
    print("creating possible mappings")
    mappings = create_possible_mappings(low_model, high_model, fixed_assignments, unwanted_low_nodes=unwanted_low_nodes)
    #print(len(mappings))
    #for mapping in mappings:
    #    print(mapping)
    print("testing mappings")
    for mapping in mappings:
    #    print(len(test_mapping(low_model, high_model, high_inputs,total_high_interventions, mapping, input_mapping).keys()))
        result.append((test_mapping(low_model, high_model, high_inputs,total_high_interventions, mapping, input_mapping),mapping))
    return result


"""
adj and adv

"""