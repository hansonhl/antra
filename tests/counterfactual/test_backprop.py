from itertools import product
from pprint import pprint

import numpy as np
from antra import *
from antra.interchange.counterfactual import CounterfactualTraining
from antra.location import location_to_str, reduce_dim

import torch
import torch.nn.functional as F
# from causal_abstraction.abstraction import find_abstractions
# import numpy as np
# from causal_abstraction.clique_analysis import construct_graph, find_cliques


from torch.utils.tensorboard import SummaryWriter

class BooleanLogicProgram(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")

        @GraphNode(leaf1,leaf2)
        def intermediate(x,y):
            return x & y

        @GraphNode(intermediate, leaf3)
        def root(w,v ):
            return w & v

        super().__init__(root)

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(2,2)
        self.lin2 = torch.nn.Linear(3,1)

    def forward(self, x,y,z):
        x1 = torch.cat([x,y], dim=-1)
        h1 = self.lin1(x1)
        h1 = F.relu(h1)
        x2 = torch.cat([h1, z], dim=-1)
        h2 = self.lin2(x2)
        return F.relu(h2)


class NeuralNetworkCompGraph(ComputationGraph):
    def __init__(self, model):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")

        self.model = model

        @GraphNode(leaf1,leaf2)
        def hidden1(x,y):
            # print(f"{x.shape=} {y.shape=}")
            x1 = torch.cat([x, y], dim=-1)
            # print(f"{a.shape=}")
            a1 = self.model.lin1(x1)
            a1.retain_grad()
            # h = torch.matmul(a, self.model.lin1.T) + self.model.bias1
            h1 = F.relu(a1)
            return h1

        @GraphNode(hidden1, leaf3)
        def root(h, z):
            # print(f"{h.shape=} {z.shape=}")
            x2 = torch.cat([h,z], dim=-1)
            h2 = self.model.lin2(x2)
            a2 = F.relu(h2)
            return a2

        super().__init__(root)


def test_equal_grad():
    # writer = SummaryWriter(log_dir="tests/interchange/counterfactual/")
    high_model= BooleanLogicProgram()
    # torch.manual_seed(42)

    for seed in np.random.randint(0, 10000, size=(4,)):
        seed = seed.item()
        torch.manual_seed(seed)

        nn = NeuralNetwork()
        low_model = NeuralNetworkCompGraph(nn)

        nn2 = NeuralNetwork()
        nn2.lin1.weight.data = nn.lin1.weight.detach().clone()
        nn2.lin1.bias.data = nn.lin1.bias.detach().clone()
        nn2.lin2.weight.data = nn.lin2.weight.detach().clone()
        nn2.lin2.bias.data = nn.lin2.bias.detach().clone()

        low_inputs = [
            GraphInput({
                "leaf1": torch.tensor([a]),
                "leaf2": torch.tensor([b]),
                "leaf3": torch.tensor([c])
            }, cache_results=False) for (a, b, c) in product((-1., 1.), repeat=3)
        ]

        high_inputs = [
            GraphInput({
                "leaf1": torch.tensor([a]),
                "leaf2": torch.tensor([b]),
                "leaf3": torch.tensor([c])
            }, cache_results=False) for (a, b, c) in product((False, True), repeat=3)
        ]

        for i, j in product(range(len(low_inputs)), repeat=2):
            li1, hi1 = low_inputs[i], high_inputs[i]
            li2, hi2 = low_inputs[j], high_inputs[j]
            low_model.model.zero_grad()
            li2_hidden = low_model.compute_node("hidden1", li2)
            low_interv = Intervention(li1, {"hidden1": li2_hidden}, cache_results=False)
            _, logits = low_model.intervene(low_interv)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            hi2_mid = high_model.compute_node("intermediate", hi2)
            hi_interv = Intervention(hi1, {"intermediate": hi2_mid}, cache_results=False)
            _, label = high_model.intervene(hi_interv)
            label = label.to(torch.float)
            loss = loss_fn(logits, label)

            loss.backward()

            lin1w_grad = low_model.model.lin1.weight.grad
            lin1b_grad = low_model.model.lin1.bias.grad
            lin2w_grad = low_model.model.lin2.weight.grad
            lin2b_grad = low_model.model.lin2.bias.grad

            nn2.zero_grad()
            x1 = torch.cat([li2["leaf1"], li2["leaf2"]], dim=-1)
            h1 = F.relu(nn2.lin1(x1))
            x2 = torch.cat([h1, li1["leaf3"]], dim=-1)
            z = F.relu(nn2.lin2(x2))
            loss2 = loss_fn(z, label)
            loss2.backward()

            assert torch.allclose(lin1w_grad, nn2.lin1.weight.grad)
            assert torch.allclose(lin1b_grad, nn2.lin1.bias.grad)
            assert torch.allclose(lin2w_grad, nn2.lin2.weight.grad)
            assert torch.allclose(lin2b_grad, nn2.lin2.bias.grad)


def test_equal_grad_cached_inputs():
    # writer = SummaryWriter(log_dir="tests/interchange/counterfactual/")

    high_model= BooleanLogicProgram()
    # torch.manual_seed(42)
    for seed in np.random.randint(0, 10000, size=(4,)):
        seed = seed.item()
        torch.manual_seed(seed)
        nn = NeuralNetwork()
        low_model = NeuralNetworkCompGraph(nn)

        nn2 = NeuralNetwork()
        nn2.lin1.weight.data = nn.lin1.weight.detach().clone()
        nn2.lin1.bias.data = nn.lin1.bias.detach().clone()
        nn2.lin2.weight.data = nn.lin2.weight.detach().clone()
        nn2.lin2.bias.data = nn.lin2.bias.detach().clone()

        low_inputs = [
            GraphInput({
                "leaf1": torch.tensor([a]),
                "leaf2": torch.tensor([b]),
                "leaf3": torch.tensor([c])
            }) for (a, b, c) in product((-1., 1.), repeat=3)
        ]

        high_inputs = [
            GraphInput({
                "leaf1": torch.tensor([a]),
                "leaf2": torch.tensor([b]),
                "leaf3": torch.tensor([c])
            }) for (a, b, c) in product((False, True), repeat=3)
        ]

        for i, j in product(range(len(low_inputs)), repeat=2):
            li1, hi1 = low_inputs[i], high_inputs[i]
            li2, hi2 = low_inputs[j], high_inputs[j]
            low_model.model.zero_grad()
            li2_hidden = low_model.compute_node("hidden1", li2)
            low_interv = Intervention(li1, {"hidden1": li2_hidden}, cache_results=False)
            _, logits = low_model.intervene(low_interv)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            hi2_mid = high_model.compute_node("intermediate", hi2)
            hi_interv = Intervention(hi1, {"intermediate": hi2_mid}, cache_results=False)
            _, label = high_model.intervene(hi_interv)
            label = label.to(torch.float)
            loss = loss_fn(logits, label)

            loss.backward(retain_graph=True) ### This actually works!!!

            lin1w_grad = low_model.model.lin1.weight.grad
            lin1b_grad = low_model.model.lin1.bias.grad
            lin2w_grad = low_model.model.lin2.weight.grad
            lin2b_grad = low_model.model.lin2.bias.grad

            nn2.zero_grad()
            x1 = torch.cat([li2["leaf1"], li2["leaf2"]], dim=-1)
            h1 = F.relu(nn2.lin1(x1))
            x2 = torch.cat([h1, li1["leaf3"]], dim=-1)
            z = F.relu(nn2.lin2(x2))
            loss2 = loss_fn(z, label)
            loss2.backward()

            assert torch.allclose(lin1w_grad, nn2.lin1.weight.grad)
            assert torch.allclose(lin1b_grad, nn2.lin1.bias.grad)
            assert torch.allclose(lin2w_grad, nn2.lin2.weight.grad)
            assert torch.allclose(lin2b_grad, nn2.lin2.bias.grad)
