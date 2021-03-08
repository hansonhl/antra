import pytest
import torch
from .utils import setup_intervention

from antra import ComputationGraph, GraphNode, GraphInput, Intervention, LOC, Location

def eq(g, input, node_name, other):
    return torch.allclose(g.compute_node(node_name, input), other)

def interv_eq(g, interv, node_name, other):
    _, res_after = g.intervene_node(node_name, interv)
    return torch.allclose(res_after, other)

@pytest.fixture
def tensor_input1():
    return GraphInput({"leaf1": torch.tensor([-2.,3.,1.,])})

@pytest.fixture
def tensor_arithmetic_graph():
    class TensorArithmeticGraph(ComputationGraph):
        def __init__(self):
            self.W1 = torch.tensor([[1.,0.,0.],[2.,2.,2.],[0.,0.,1.]])
            self.b1 = torch.tensor([10., 10., 10.])
            self.W2 = torch.tensor([[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]])
            self.b2 = torch.tensor([-2., -2., -2.])

            @GraphNode()
            def leaf1(x):
                return self.leaf1func(x)

            @GraphNode(leaf1)
            def h1(l1):
                return self.h1func(l1)

            @GraphNode(leaf1)
            def h2(l1):
                return self.h2func(l1)

            @GraphNode(h1, h2)
            def add(x, y):
                return self.addfunc(x, y)

            @GraphNode(add)
            def relu(a):
                return self.relufunc(a)

            @GraphNode(relu)
            def root(a):
                return self.sumfunc(a)

            super(TensorArithmeticGraph, self).__init__(root)

        def leaf1func(self, x):
            return x * 2

        def h1func(self, l1):
            return torch.matmul(self.W1, l1) + self.b1

        def h2func(self, l1):
            return torch.matmul(self.W2, l1) + self.b2

        def addfunc(self, h1, h2):
            return h1 + h2

        def relufunc(self, a):
            return torch.max(a, torch.zeros(3))

        def sumfunc(self, a):
            return a.sum()

    return TensorArithmeticGraph()

@pytest.mark.parametrize("x",[torch.tensor([-2.,-4.,-6.]),torch.tensor([0.,0.,0.]),
                              torch.tensor([0.,1.,0.])])
def test_tensor_arithmetic(x, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    l1 = x * 2.
    h1 = torch.matmul(g.W1, l1) + g.b1
    h2 = torch.matmul(g.W2, l1) + g.b2
    a = h1 + h2
    relu = torch.max(a, torch.zeros(3))
    expected = relu.sum()
    graph_input = GraphInput({"leaf1": x})

    graph_input2 = GraphInput({"leaf1": x})

    assert g.compute(graph_input) == expected
    assert eq(g, graph_input, "relu", relu)
    assert eq(g, graph_input, "add", a)
    assert eq(g, graph_input, "h1", h1)
    assert eq(g, graph_input, "h2", h2)

    assert g.compute(graph_input2) == expected
    assert eq(g, graph_input2, "relu", relu)
    assert eq(g, graph_input2, "add", a)
    assert eq(g, graph_input2, "h1", h1)
    assert eq(g, graph_input2, "h2", h2)


def test_tensor_arithmetic_interv1(setup_intervention, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph
    input_dict = {"leaf1": torch.tensor([-2.,3.,1.,])}
    tensor_input1 = GraphInput(input_dict)
    vanilla = g.compute(tensor_input1)

    assert eq(g, tensor_input1, "h2", torch.tensor([2., -2., 2.]))

    interv_dict = {"h2": torch.tensor([-100., -100., -100.])}

    i = setup_intervention(input_dict, interv_dict)

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h2", "add", "relu", "root"}
    assert before == 38., after == 0.
    assert before == vanilla


# def test_tensor_arithmetic_interv2(setup_intervention, tensor_arithmetic_graph):
#     g = tensor_arithmetic_graph
#
#     input_dict = {"leaf1": torch.tensor([-2., 3., 1., ])}
#     interv_dict = {"h2": torch.tensor([-10.])}
#     loc_dict = {"h2": 1}
#
#     i = setup_intervention(input_dict, interv_dict, loc_dict)
#
#     before, after = g.intervene(i)
#
#     assert i.affected_nodes == {"h2", "add", "relu", "root"}
#     assert before == 38., after == 30.
#     assert interv_eq(g, i, "h2", torch.tensor([2., -10., 2.]))
#     assert eq(g, i.base, "h2", torch.tensor([2., -2., 2.]))


def test_tensor_arithmetic_interv3(setup_intervention, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    input_dict = {"leaf1": torch.tensor([-2., 3., 1., ])}
    interv_dict = {"h2": torch.tensor([-10., -10.])}
    loc_dict = {"h2": LOC[1:]}

    i = setup_intervention(input_dict, interv_dict, loc_dict)

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h2", "add", "relu", "root"}
    assert before == 38. and after == 18.
    assert interv_eq(g, i, "h2", torch.tensor([2., -10., -10.]))

