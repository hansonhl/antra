import pytest

from antra import ComputationGraph, GraphNode, GraphInput, Intervention


@pytest.fixture
def input_leaf1():
    return GraphInput({"a": 1, "b": 10, "c": 100})


@pytest.fixture
def input_arith1():
    return GraphInput({"a": 1, "b": 10, "c": 100, "d": 1000, "e": 1000})


@pytest.fixture
def arithmetic_graph1():
    class ArithmeticGraph1(ComputationGraph):
        def __init__(self):
            a = GraphNode.leaf("a")
            b = GraphNode.leaf("b")
            c = GraphNode.leaf("c")
            d = GraphNode.leaf("d")
            e = GraphNode.leaf("e")

            @GraphNode(a, b, c)
            def node1(a, b, c):
                print("node1 = a + b + c = %d" % (a + b + c))
                return a + b + c

            @GraphNode(d, e)
            def node2(d, e):
                print("node2 = (d + e) / 10 = %f" % ((d + e) / 10))
                return (d + e) / 10

            @GraphNode(node1)
            def child1(x):
                print("child1 = node1 * 2 = %d" % (x * 2))
                return x * 2

            @GraphNode(node1, node2)
            def child2(x, y):
                print("child2 = node1 - node2 = %f" % (x - y))
                return x - y

            @GraphNode(child1, child2)
            def root(w, z):
                print("root = child1 + child2 + 1 = %f" % (w + z + 1))
                return w + z + 1

            super().__init__(root)

    return ArithmeticGraph1()


@pytest.fixture
def arithmetic_graph2():
    class ArithmeticGraph2(ComputationGraph):
        def __init__(self):
            a = GraphNode.leaf("a")
            b = GraphNode.leaf("b")
            c = GraphNode.leaf("c")

            @GraphNode(a, b, c)
            def node1(a, b, c):
                return a + b + c

            @GraphNode(node1)
            def child1(l1):
                return l1 * 2
            
            @GraphNode(child1)
            def branch1(c1):
                return c1 + 1
            
            @GraphNode(branch1)
            def branch2(b1):
                return b1 + 1
            
            @GraphNode(branch2)
            def branch3(b2):
                return b2 + 1

            @GraphNode(child1, branch3)
            def child2(c1, b3):
                return c1 + b3
            
            @GraphNode(child2, node1)
            def root(c2, l1):
                return c2 - 100 + l1
            
            super(ArithmeticGraph2, self).__init__(root)

    return ArithmeticGraph2()


@pytest.fixture
def singleton_graph():
    class SingletonGraph(ComputationGraph):
        def __init__(self):
            a = GraphNode.leaf("a")
            b = GraphNode.leaf("b")
            c = GraphNode.leaf("c")

            @GraphNode(a, b, c)
            def node1(a, b, c):
                return a + b + c

            super(SingletonGraph, self).__init__(node1)

    return SingletonGraph()


@pytest.fixture
def intervened_graph(input_leaf1, singleton_graph):
    interv = Intervention(input_leaf1)
    interv.set_intervention("node1", 100)
    _, _ = singleton_graph.intervene(interv)

    return singleton_graph


def test_singleton(input_leaf1, singleton_graph):
    assert singleton_graph.compute(input_leaf1) == 111
    assert singleton_graph.compute_node("node1", input_leaf1) == 111


def test_singleton_interv(singleton_graph, input_leaf1):
    """Intervene at root position"""
    interv = Intervention(input_leaf1)
    interv.set_intervention("node1", 100)

    before, after = singleton_graph.intervene(interv)

    assert before == 111 and after == 100
    interv_before, interv_after = singleton_graph.intervene_node("node1", interv)
    assert interv_after == 100

    assert "node1" in interv.affected_nodes and len(interv.affected_nodes) == 1


def test_singleton_clear_cache(intervened_graph):
    intervened_graph.clear_caches()

    leaf1_node = intervened_graph.nodes["node1"]
    assert len(leaf1_node.base_cache) == 0


def test_arithmetic_graph1(arithmetic_graph1, input_arith1):
    assert arithmetic_graph1.compute(input_arith1) == 134
    assert arithmetic_graph1.compute_node("child1", input_arith1) == 222
    assert arithmetic_graph1.compute_node("child2", input_arith1) == -89
    assert arithmetic_graph1.compute_node("node1", input_arith1) == 111
    assert arithmetic_graph1.compute_node("node2", input_arith1) == 200


def test_arithmetic_graph1_interv(arithmetic_graph1, input_arith1):
    i = Intervention(input_arith1)
    i.set_intervention("node1", 100)

    before, after = arithmetic_graph1.intervene(i)

    assert i.affected_nodes == {"node1", "child1", "child2", "root"}
    assert before == 134 and after == 101

    assert arithmetic_graph1.compute_node("child1", input_arith1) == 222
    assert arithmetic_graph1.compute_node("child2", input_arith1) == -89
    assert arithmetic_graph1.compute_node("node1", input_arith1) == 111
    assert arithmetic_graph1.compute_node("node2", input_arith1) == 200

    assert arithmetic_graph1.intervene_node("child1", i)[1] == 200
    assert arithmetic_graph1.intervene_node("child2", i)[1] == -100
    assert arithmetic_graph1.intervene_node("node1", i)[1] == 100
    assert arithmetic_graph1.intervene_node("node2", i)[1] == 200


def test_arithmetic_graph2(arithmetic_graph2, input_leaf1):
    assert arithmetic_graph2.compute(input_leaf1) == 458

    assert arithmetic_graph2.compute_node("child2", input_leaf1) == 447
    assert arithmetic_graph2.compute_node("child1", input_leaf1) == 222
    assert arithmetic_graph2.compute_node("branch3", input_leaf1) == 225


def test_arithmetic_graph2_interv(arithmetic_graph2, input_leaf1):
    i = Intervention(input_leaf1)
    i.set_intervention("child1", 200)

    before, after = arithmetic_graph2.intervene(i)

    assert i.affected_nodes == {"child1", "branch1", "branch2", "branch3",
                                "child2", "root"}
    assert before == 458, after == 414

    assert arithmetic_graph2.compute_node("child2", input_leaf1) == 447
    assert arithmetic_graph2.compute_node("child1", input_leaf1) == 222
    assert arithmetic_graph2.compute_node("branch3", input_leaf1) == 225
    assert arithmetic_graph2.compute_node("node1", input_leaf1) == 111


    assert arithmetic_graph2.intervene_node("child2", i)[1] == 403
    assert arithmetic_graph2.intervene_node("child1", i)[1] == 200
    assert arithmetic_graph2.intervene_node("branch3", i)[1] == 203
    assert arithmetic_graph2.intervene_node("node1", i)[1] == 111

# TODO: Test multiple sites of intervention

def test_empty_interv(arithmetic_graph2, input_leaf1):
    i = Intervention(input_leaf1)

    with pytest.raises(RuntimeError) as excinfo:
        _, _ = arithmetic_graph2.intervene(i)
        assert "Must specify" in str(excinfo)
