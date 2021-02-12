import torch
import pytest
from compgraph import CompGraphConstructor


def has_child(g, parent_node, child_node):
    return child_node in g.nodes[parent_node].children_dict


def has_children(g, parent_node, child_nodes):
    return set(child_nodes) == set(g.nodes[parent_node].children_dict.keys())


def is_leaf(g, node):
    return len(g.nodes[node].children) == 0


class TorchEqualityModule(torch.nn.Module):
    def __init__(self,
                 input_size=20,
                 hidden_layer_size=100,
                 activation="relu"):
        super(TorchEqualityModule, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_layer_size)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise NotImplementedError("Activation method not implemented")
        self.linear2 = torch.nn.Linear(hidden_layer_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        linear_out = self.linear1(x)
        hidden = self.activation(linear_out)
        logits = self.linear2(hidden)
        return self.sigmoid(logits)


class Residual(torch.nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

    def forward(self, x, y):
        return x + y


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim=16):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.residual = Residual()

    def forward(self, x):
        out = self.linear(x)
        out = self.residual(out, x)
        out = self.relu(out)
        return out


class FFNN(torch.nn.Module):
    def __init__(self, hidden_dim=16, num_layers=4, num_classes=4):
        super(FFNN, self).__init__()
        self.mlps = torch.nn.ModuleList(MLP(hidden_dim)
                                        for _ in range(num_layers))
        self.logits = torch.nn.Linear(hidden_dim, num_classes)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)

        logits = self.logits(x)
        return self.softmax(logits)


@pytest.fixture()
def ffnn1():
    return FFNN()


@pytest.fixture()
def mlp1():
    return TorchEqualityModule()


def test_mlp(mlp1):
    x = torch.randn(20)
    g, x = CompGraphConstructor.construct(mlp1, x)

    node_names = {"linear1", "linear2", "activation", "sigmoid"}
    assert set(g.nodes.keys()) == node_names
    assert is_leaf(g, "linear1")
    assert has_children(g, "activation", ["linear1"])
    assert has_children(g, "linear2", ["activation"])
    assert has_children(g, "sigmoid", ["linear2"])


def test_ffnn(ffnn1):
    x = torch.randn(16)
    submodules = {"logits": ffnn1.logits,
                  "softmax": ffnn1.softmax}
    submodules.update({"mlp" + i: m for i, m in ffnn1.mlps.named_children()})
    del submodules["mlp1"]
    submodules.update({"mlp1_" + n: m for n, m in
                       ffnn1.mlps[1].named_children()})

    node_names = {"mlp0", "mlp1_linear", "mlp1_relu", "mlp1_residual",
                  "mlp2", "mlp3", "logits", "softmax"}

    g, in_x = CompGraphConstructor.construct(ffnn1, x, submodules=submodules)

    assert node_names == set(g.nodes.keys())
    assert is_leaf(g, "mlp0")
    assert has_children(g, "mlp1_linear", ["mlp0"])
    assert has_children(g, "mlp1_residual", ["mlp0", "mlp1_linear"])
    assert has_children(g, "mlp1_relu", ["mlp1_residual"])
    assert has_children(g, "mlp2", ["mlp1_relu"])
    assert has_children(g, "mlp3", ["mlp2"])
    assert has_children(g, "logits", ["mlp3"])
    assert has_children(g, "softmax", ["logits"])

    _ = g.compute(in_x)
