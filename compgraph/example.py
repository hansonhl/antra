import torch
from intervention import *


if __name__ == "__main__":
    ##### Example 1 #####
    class MyCompGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf1(a, b, c):
                print("leaf1 = a + b + c = %d" % (a + b + c))
                return a + b + c

            @GraphNode()
            def leaf2(d, e):
                print("leaf2 = (d + e) / 10 = %f" % ((d + e) / 10))
                return (d + e) / 10

            @GraphNode(leaf1)
            def child1(x):
                print("child1 = leaf1 * 2 = %d" % (x * 2))
                return x * 2

            @GraphNode(leaf1, leaf2)
            def child2(x, y):
                print("child2 = leaf1 - leaf2 = %f" % (x - y))
                return x - y

            @GraphNode(child1, child2)
            def root(w, z):
                print("root = child1 + child2 + 1 = %f" % (w + z + 1))
                return w + z + 1

            super().__init__(root)


    print("----- Example 1  -----")

    g = MyCompGraph()
    g.clear_caches()

    interv = GraphInput({"leaf1": (10, 20, 30), "leaf2": (2, 3)})
    in1 = Intervention(interv, {"child1": 100})

    res = g.intervene(in1)
    print(res)


    ##### Example 2 ######
    class Graph2(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf1(x, y):
                return x + y

            @GraphNode(leaf1)
            def leaf2(z):
                return -1 * z

            @GraphNode(leaf2)
            def root(x):
                return x.sum()

            super().__init__(root)


    print("----- Example 2  -----")

    g2 = Graph2()
    g.clear_caches()

    i1 = GraphInput(
        {"leaf1": (torch.tensor([10, 20, 30]), torch.tensor([1, 1, 1]))})
    in1 = Intervention(i1)
    in1["leaf2[:2]"] = torch.tensor([101, 201])

    before, after = g2.intervene(in1)
    print("Before:", before, "after:", after)

    interv = {"leaf1": torch.tensor([300, 300]), "leaf2": torch.tensor([100])}
    locs = {"leaf1": Location()[:2], "leaf2": 2}
    in2 = Intervention(i1, intervention=interv, location=locs)
    before, after = g2.intervene(in2)
    print("Before:", before, "after:", after)


    ##### Example 3 #####

    class TorchEqualityModule(torch.nn.Module):
        def __init__(self,
                     input_size=20,
                     hidden_layer_size=100,
                     activation="relu"):
            super(TorchEqualityModule, self).__init__()
            self.linear = torch.nn.Linear(input_size, hidden_layer_size)
            if activation == "relu":
                self.activation = torch.nn.ReLU()
            else:
                raise NotImplementedError("Activation method not implemented")
            self.output = torch.nn.Linear(hidden_layer_size, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            linear_out = self.linear(x)
            self.hidden_vec = self.activation(linear_out)
            logits = self.output(self.hidden_vec)
            return self.sigmoid(logits)


    module = TorchEqualityModule()
    input = torch.randn(20)
    g3, in3 = CompGraphConstructor.construct(module, input)
    print("----- Example 3 -----")
    print("Nodes of graph:", ", ".join(g3.nodes))
    print("Name of root:", g3.root.name)
