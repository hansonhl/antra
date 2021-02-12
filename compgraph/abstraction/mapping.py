import copy


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