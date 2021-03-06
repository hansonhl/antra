import torch
from antra import *


if __name__ == "__main__":
    ##### Example 1 #####
    class SimpleCompGraph(ComputationGraph):
        def __init__(self):
            a = GraphNode.leaf("a")
            b = GraphNode.leaf("b")
            c = GraphNode.leaf("c")
            d = GraphNode.leaf("d")
            e = GraphNode.leaf("e")

            @GraphNode(a, b, c)
            def node1(a, b, c):
                print(f"node1 = a + b + c = {a+b+c}")
                return a + b + c

            @GraphNode(d, e)
            def node2(d, e):
                print(f"node2 = (d + e) / 10 = {((d + e) / 10)}")
                return (d + e) / 10

            @GraphNode(node1)
            def node3(x):
                print(f"node3 = node1 * 2 = {(x * 2)}")
                return x * 2

            @GraphNode(node1, node2)
            def node4(x, y):
                print(f"node4 = node1 - node2 = {(x - y)}")
                return x - y

            @GraphNode(node3, node4)
            def root(w, z):
                print(f"root = node3 + node4 + 1 = {(w + z + 1)}")
                return w + z + 1

            super().__init__(root)


    print("----- Example 1  -----")

    g = SimpleCompGraph()
    g.clear_caches()

    input1 = GraphInput({"a":10, "b": 20, "c": 30, "d": 2, "e": 3})
    interv1 = Intervention(input1, {"node3": 100})

    before, after = g.intervene(interv1)
    print(f"Before: {before}, After: {after}")


    ##### Example 2 ######
    class GraphWithTensors(ComputationGraph):
        def __init__(self):
            x = GraphNode.leaf("x")
            y = GraphNode.leaf("y")

            @GraphNode(x, y)
            def node1(x, y):
                return x + y

            print(node1.children)

            @GraphNode(node1)
            def node2(z):
                return -1 * z

            @GraphNode(node2)
            def root(x):
                return x.sum(dim=-1) # to support batch summation

            super().__init__(root)


    print("----- Example 2  -----")

    g2 = GraphWithTensors()
    g2.clear_caches()

    input2 = GraphInput({"x": torch.tensor([10, 20, 30]),
                        "y": torch.tensor([1, 1, 1])})
    interv2 = Intervention(input2)
    interv2["node2[:2]"] = torch.tensor([101, 201])

    before, after = g2.intervene(interv2)
    print("Before:", before, "after:", after)

    node1_res = g2.compute_node("node1", input2)
    print("node1 result", node1_res)

    input3 = {"node1": torch.tensor([300, 300]), "node2": torch.tensor([100])}
    locs = {"node1": LOC[:2], "node2": 2}
    interv3 = Intervention(input2, intervention=input3, location=locs)
    before, after = g2.intervene(interv3)
    print("Before:", before, "after:", after)

    print("----- Batched inputs -----")

    input_x_batch = torch.tensor([[10, 20, 30],
                               [10, 20, 30],
                               [10, 20, 30],
                               [100, 200, 300],
                               [100, 200, 300]])
    input_y_batch = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1,],
                                  [10, 10, 10], [10, 20, 30]])

    input_batch = GraphInput.batched({"x": input_x_batch, "y": input_y_batch})

    interv_input = torch.tensor([[100, 100]] * 5)
    interv_batch = Intervention.batched(input_batch, intervention={"node1[:,:2]": interv_input})
    # notes: location must match batched format

    before, after = g2.intervene(interv_batch)
    print("Before:", before, "after:", after)

