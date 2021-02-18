import pytest
from antra.abstractable import AbstractableCompGraph

graph_1 = {"A": ["B", "C"],
           "B": ["E"],
           "C": ["D"],
           "D": ["E"],
           "E": []}
expected_1 = ["A", "B", "C", "D", "E"]

graph_2 = {"A": ["B", "C", "D"],
           "B": ["E"],
           "C": ["E", "F"],
           "D": ["F"],
           "E": ["G", "H"],
           "F": ["H"],
           "G": [],
           "H": []}

expected_2 = ["A", "B", "C", "E", "G", "D", "F", "H"]


test_set = [(graph_1, "A", expected_1),
            (graph_2, "A", expected_2)]

@pytest.mark.parametrize("full_graph,root,expected", test_set)
def test_find_topological_order(full_graph, root, expected):
    order = AbstractableCompGraph.find_topological_order(full_graph, root)
    assert all(n1 == n2 for n1, n2 in zip(order, expected)), f"expected f{expected}, got f{order}"


