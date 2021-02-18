import pytest
import torch

from antra import ComputationGraph, GraphNode, GraphInput, Intervention, LOC

def eq(g, input, node_name, other):
    return torch.all(g.compute_node(node_name, input) == other)

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
                return x * 2.

            @GraphNode(leaf1)
            def h1(l1):
                return torch.matmul(self.W1, l1) + self.b1

            @GraphNode(leaf1)
            def h2(l1):
                return torch.matmul(self.W2, l1) + self.b2

            @GraphNode(h1, h2)
            def add(x, y):
                return x + y

            @GraphNode(add)
            def relu(a):
                return torch.max(a, torch.zeros(3))

            @GraphNode(relu)
            def root(a):
                return a.sum()

            super(TensorArithmeticGraph, self).__init__(root)
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

def test_tensor_arithmetic_interv1(tensor_input1, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    vanilla = g.compute(tensor_input1)

    assert eq(g, tensor_input1, "h2", torch.tensor([2., -2., 2.]))

    i = Intervention(tensor_input1)
    i.set_intervention("h2", torch.tensor([-100., -100., -100.]))

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h2", "add", "relu", "root"}
    assert before == 38., after == 0.
    assert before == vanilla


def test_tensor_arithmetic_interv2(tensor_input1, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    i = Intervention(tensor_input1)
    i.set_intervention("h2[1]", torch.tensor([-10.]))

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h2", "add", "relu", "root"}
    assert before == 38., after == 30.
    assert eq(g, i, "h2", torch.tensor([2., -10., 2.]))
    assert eq(g, i.base, "h2", torch.tensor([2., -2., 2.]))


def test_tensor_arithmetic_interv3(tensor_input1, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    i = Intervention(tensor_input1)
    i.set_intervention("h2", torch.tensor([-10., -10.]))
    i.set_location("h2", LOC[1:])

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h2", "add", "relu", "root"}
    assert before == 38. and after == 18.
    assert eq(g, i, "h2", torch.tensor([2., -10., -10.]))


def test_multiple_interv(tensor_input1, tensor_arithmetic_graph):
    g = tensor_arithmetic_graph

    i = Intervention(tensor_input1)
    i.set_intervention("h1[0]", torch.tensor([-6.]))
    i.set_intervention("h2[1]", torch.tensor([-10.]))

    before, after = g.intervene(i)

    assert i.affected_nodes == {"h1", "h2", "add", "relu", "root"}
    assert before == 38. and after == 22.
    assert eq(g, i, "h1", torch.tensor([-6., 18., 12.]))
    assert eq(g, i, "h2", torch.tensor([2., -10., 2.]))
    assert eq(g, i.base, "h1", torch.tensor([6., 18., 12.]))
    assert eq(g, i.base, "h2", torch.tensor([2., -2., 2.]))