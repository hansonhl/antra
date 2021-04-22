import pytest
import torch
import torch.nn.functional as F

from antra import *
from antra.abstractable import AbstractableCompGraph, find_topological_order

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
    order = find_topological_order(full_graph, root)
    assert all(n1 == n2 for n1, n2 in zip(order, expected)), f"expected f{expected}, got f{order}"

batch_size = 10

class TensorArithmeticGraphDim0(ComputationGraph):
    def __init__(self):
        self.W1 = torch.tensor([[1.,0.,0.],[2.,2.,2.],[0.,0.,1.]])
        self.b1 = torch.tensor([10., 10., 10.])
        self.W2 = torch.tensor([[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]])
        self.b2 = torch.tensor([-2., -2., -2.])

        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")

        @GraphNode(leaf1)
        def h1(l1):
            return self.h1_func(l1)

        @GraphNode(leaf2)
        def h2(l2):
            return self.h2_func(l2)

        @GraphNode(h1, h2)
        def add(x, y):
            return self.add_func(x, y)

        @GraphNode(add)
        def relu(a):
            return self.relu_func(a)

        @GraphNode(relu)
        def root(a):
            return self.root_func(a)

        super(TensorArithmeticGraphDim0, self).__init__(root)

    def h1_func(self, l1):
        return torch.matmul(l1, self.W1) + self.b1

    def h2_func(self, l1):
        return torch.matmul(l1, self.W2) + self.b2

    def add_func(self, x, y):
        return x + y

    def relu_func(self, a):
        return F.relu(a)

    def root_func(self, a):
        return a.sum(dim=1)

def test_abstractable_structure():
    g = TensorArithmeticGraphDim0()
    ag = AbstractableCompGraph(g, ["add", "relu"])
    expected_nodes = ["leaf1", "leaf2", "add", "relu", "root"]
    assert len(ag.nodes) == len(expected_nodes)
    assert all(node_name in ag.nodes for node_name in expected_nodes)

def test_abstractable_forward_computation():
    g = TensorArithmeticGraphDim0()
    ag = AbstractableCompGraph(g, ["add", "relu"])

    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        batched_input = GraphInput.batched({"leaf1": input1, "leaf2": input2})
        res = ag.compute(batched_input)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        expected = g.root_func(r)

        assert torch.allclose(res, expected)
