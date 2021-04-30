import pytest
import torch
import torch.nn.functional as F
from antra import ComputationGraph, GraphNode, GraphInput, Intervention, Location, LOC
import itertools

from pprint import pprint
from .utils import setup_intervention, setup_intervention_func_for_fixture

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

batch_size = 20

def test_batch_base_computation_torch_dim0():
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        batched_input = GraphInput.batched({"leaf1": input1, "leaf2": input2})
        res = g.compute(batched_input)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        expected = g.root_func(r)

        assert torch.allclose(res, expected)

def test_batch_interv_keys_0():
    input1 = torch.tensor([[0., 1., 2.,], [10., 11., 12.,], [20., 21., 22.]])
    input2 = torch.tensor([[100., 101., 102.], [110., 111., 112.,], [120., 121., 122.]])
    input_values = GraphInput.batched({"leaf1": input1, "leaf2": input2})
    interv_values = {"h2[:2]": torch.ones(3, 2)}
    batched_interv = Intervention.batched(input_values, interv_values)

    for ex, i1, i2 in zip(batched_interv.keys, input1, input2):
        base_key, interv_key = ex
        assert len(base_key) == 2
        assert base_key[0][0] == "leaf1" and base_key[1][0] == "leaf2"
        assert torch.allclose(torch.tensor(base_key[0][1]), i1)
        assert torch.allclose(torch.tensor(base_key[1][1]), i2)
        assert torch.allclose(torch.tensor(interv_key[0][1]), torch.tensor([1.,1.]))
        assert interv_key[0][0] == "h2[:2:]"

    base_keys = batched_interv.base.keys
    for base_key, i1, i2 in zip(base_keys, input1, input2):
        assert len(base_key) == 2
        assert base_key[0][0] == "leaf1" and base_key[1][0] == "leaf2"
        assert torch.allclose(torch.tensor(base_key[0][1]), i1)
        assert torch.allclose(torch.tensor(base_key[1][1]), i2)


def test_batch_one_interv_dim0_0():
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        input_values = GraphInput.batched({"leaf1": input1, "leaf2": input2})

        interv_input = torch.ones(batch_size, 3)
        interv_values = {"h2": interv_input}

        batched_interv = Intervention.batched(input_values, interv_values)

        base_res, interv_res = g.intervene(batched_interv)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        base_expected = g.root_func(r)

        assert torch.allclose(base_res, base_expected)

        a_prime = g.add_func(h1, interv_input)
        r_prime = g.relu_func(a_prime)
        interv_expected = g.root_func(r_prime)

        h1_before, h1_after = g.intervene_node("h1", batched_interv)
        add_before, add_after = g.intervene_node("add", batched_interv)
        r_before, r_after = g.intervene_node("relu", batched_interv)

        assert torch.allclose(h1, h1_after)
        assert torch.allclose(a_prime, add_after)
        assert torch.allclose(r_prime, r_after)

        assert torch.allclose(interv_res, interv_expected)



def test_batch_one_interv_dim0_1(setup_intervention):
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        interv = torch.ones(batch_size, 2)

        input_dict = {"leaf1": input1, "leaf2": input2}
        interv_dict = {"h2": interv}
        loc_dict = {"h2": LOC[:, 1:]}

        batched_interv = setup_intervention(input_dict, interv_dict, loc_dict, batched=True, batch_dim=0)
        # input_values = {"leaf1": input1, "leaf2": input2}
        #
        # interv_input = torch.ones(batch_size, 2)
        # interv_values = {"h2[:,1:]": interv_input}
        #
        # batched_interv = Intervention.batched(input_values, interv_values)

        base_res, interv_res = g.intervene(batched_interv)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        base_expected = g.root_func(r)

        assert torch.allclose(base_res, base_expected)

        h2_prime = h2.detach().clone()
        h2_prime[:,1:] = interv
        a_prime = g.add_func(h1, h2_prime)
        r_prime = g.relu_func(a_prime)
        interv_expected = g.root_func(r_prime)

        h1_before, h1_after = g.intervene_node("h1", batched_interv)
        add_before, add_after = g.intervene_node("add", batched_interv)
        r_before, r_after = g.intervene_node("relu", batched_interv)

        assert torch.allclose(h1, h1_after)
        assert torch.allclose(h1, h1_before)
        assert torch.allclose(a_prime, add_after)
        assert torch.allclose(a, add_before)
        assert torch.allclose(r_prime, r_after)
        assert torch.allclose(r, r_before)
        assert torch.allclose(interv_res, interv_expected)

class TensorArithmeticGraphDim1(ComputationGraph):
    def __init__(self):
        self.W1 = torch.tensor([[1.,0.,0.],[2.,2.,2.],[0.,0.,1.]])
        self.b1 = torch.tensor([10., 10., 10.]).unsqueeze(1)
        self.W2 = torch.tensor([[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]])
        self.b2 = torch.tensor([-2., -2., -2.]).unsqueeze(1)

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

        super(TensorArithmeticGraphDim1, self).__init__(root)

    def h1_func(self, l1):
        return torch.matmul(self.W1, l1) + self.b1

    def h2_func(self, l1):
        return torch.matmul(self.W2, l1) + self.b2

    def add_func(self, x, y):
        return x + y

    def relu_func(self, a):
        return F.relu(a)

    def root_func(self, a):
        return a.sum(dim=0)


def test_batch_one_interv_dim1(setup_intervention):
    g = TensorArithmeticGraphDim1()
    torch.manual_seed(39)
    inputs1 = torch.randn(3, 1000)
    inputs2 = torch.randn(3, 1000) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[:, i:i+batch_size]
        input2 = inputs2[:, i:i+batch_size]
        interv = torch.ones(2, batch_size)

        input_dict = {"leaf1": input1, "leaf2": input2}
        interv_dict = {"h2": interv}
        loc_dict = {"h2": LOC[1:]}

        batched_interv = setup_intervention(input_dict, interv_dict, loc_dict,
                                            batched=True, batch_dim=1)

        # input_values = {"leaf1": input1, "leaf2": input2}
        # interv_values = {"h2[1:]": interv}
        # batched_interv = Intervention.batched(input_values, interv_values, batch_dim=1)
        base_res, interv_res = g.intervene(batched_interv)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        base_expected = g.root_func(r)

        assert torch.allclose(base_res, base_expected)

        h2_prime = h2.detach().clone()
        h2_prime[1:] = interv
        a_prime = g.add_func(h1, h2_prime)
        r_prime = g.relu_func(a_prime)
        interv_expected = g.root_func(r_prime)

        h1_before, h1_after = g.intervene_node("h1", batched_interv)
        add_before, add_after = g.intervene_node("add", batched_interv)
        r_before, r_after = g.intervene_node("relu", batched_interv)

        assert torch.allclose(h1, h1_after)
        assert torch.allclose(h1, h1_before)
        assert torch.allclose(a_prime, add_after)
        assert torch.allclose(a, add_before)
        assert torch.allclose(r_prime, r_after)
        assert torch.allclose(r, r_before)
        assert torch.allclose(interv_res, interv_expected)

# Only limited ways are allowed
interv_construction_types = [("dict", "GraphInput"),
                             ("dict",),
                             ("dict", "interv_str",)]
params = [t for t in itertools.product(*interv_construction_types)]
params += [("dict", "set", "interv_str"), ("GraphInput", "set", "interv_str")]
idfn = lambda t: "/".join(t)
@pytest.fixture(params=params, ids=idfn)
def setup_multi_loc_batched_intervention(request):
    return setup_intervention_func_for_fixture(request)

def test_batch_one_interv_multi_locs(setup_multi_loc_batched_intervention):
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        interv1 = torch.ones(batch_size)
        interv2 = torch.zeros(batch_size)

        input_dict = {"leaf1": input1, "leaf2": input2}
        interv_dict = {"h2": [interv1, interv2]}
        loc_dict = {"h2": [LOC[:,0], LOC[:,2]]}

        batched_interv = setup_multi_loc_batched_intervention(input_dict, interv_dict, loc_dict,
                                            batched=True, batch_dim=0)

        base_res, interv_res = g.intervene(batched_interv)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        base_expected = g.root_func(r)

        h2_after = h2.detach().clone()
        h2_after[:,0] = interv1
        h2_after[:,2] = interv2
        a_prime = g.add_func(h1, h2_after)
        r_prime = g.relu_func(a_prime)
        interv_expected = g.root_func(r_prime)

        h1_before, h1_after = g.intervene_node("h1", batched_interv)
        a_before, a_after = g.intervene_node("add", batched_interv)
        r_before, r_after = g.intervene_node("relu", batched_interv)

        eq_pairs = [(h1, h1_after), (h1, h1_before), (a_prime, a_after),
                    (a, a_before), (r, r_before), (r_prime, r_after),
                    (interv_res, interv_expected), (base_res, base_expected)]

        for x, y in eq_pairs:
            assert torch.allclose(x, y)


# Only limited ways are allowed
interv_construction_types = [("dict", "GraphInput"),
                             ("dict",),
                             ("dict", "interv_str",)]
params = [t for t in itertools.product(*interv_construction_types)]
params += [("dict", "set", "interv_str"), ("GraphInput", "set", "interv_str")]
idfn = lambda t: "/".join(t)
@pytest.fixture(params=params, ids=idfn)
def setup_multi_loc_batched_intervention(request):
    return setup_intervention_func_for_fixture(request)

def test_batch_one_interv_multi_locs_intervene_all(setup_multi_loc_batched_intervention):
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for i in range(0, 1000, batch_size):
        input1 = inputs1[i:i+batch_size]
        input2 = inputs2[i:i+batch_size]
        interv1 = torch.ones(batch_size)
        interv2 = torch.zeros(batch_size)

        input_dict = {"leaf1": input1, "leaf2": input2}
        interv_dict = {"h2": [interv1, interv2]}
        loc_dict = {"h2": [LOC[:,0], LOC[:,2]]}

        batched_interv = setup_multi_loc_batched_intervention(
            input_dict, interv_dict, loc_dict, batched=True, batch_dim=0)

        base_res, interv_res = g.intervene(batched_interv)

        h1 = g.h1_func(input1)
        h2 = g.h2_func(input2)
        a = g.add_func(h1, h2)
        r = g.relu_func(a)
        base_expected = g.root_func(r)

        h2_after = h2.detach().clone()
        h2_after[:,0] = interv1
        h2_after[:,2] = interv2
        a_prime = g.add_func(h1, h2_after)
        r_prime = g.relu_func(a_prime)
        interv_expected = g.root_func(r_prime)

        base_res_dict, ivn_res_dict = g.intervene_all_nodes(batched_interv)

        # pprint(base_res_dict)
        # pprint(ivn_res_dict)
        # break

        h1_before, h1_after = base_res_dict["h1"], ivn_res_dict["h1"]
        a_before, a_after = base_res_dict["add"], ivn_res_dict["add"]
        r_before, r_after = base_res_dict["relu"], ivn_res_dict["relu"]

        eq_pairs = [(h1, h1_after), (h1, h1_before), (a_prime, a_after),
                    (a, a_before), (r, r_before), (r_prime, r_after),
                    (interv_res, interv_expected), (base_res, base_expected)]

        for x, y in eq_pairs:
            assert torch.allclose(x, y)


def test_batch_multi_interv_multi_locs(setup_multi_loc_batched_intervention):
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    inputs1 = torch.randn(1000, 3)
    inputs2 = torch.randn(1000, 3) * 2.
    for _ in range(3):
        for i in range(0, 1000, batch_size):
            with torch.no_grad():
                input1 = inputs1[i:i + batch_size]
                input2 = inputs2[i:i + batch_size]
                interv1 = torch.ones(batch_size)
                interv2 = torch.zeros(batch_size)
                interv3 = torch.ones(batch_size)

                input_dict = {"leaf1": input1, "leaf2": input2}
                interv_dict = {"h2": [interv1, interv2], "add": interv3}
                loc_dict = {"h2": [LOC[:, 0], LOC[:, 2]],
                            "add": LOC[:, 1]}

                batched_interv = setup_multi_loc_batched_intervention(input_dict, interv_dict, loc_dict, batched=True, batch_dim=0)

                base_res, interv_res = g.intervene(batched_interv)
                print("affected nodes", batched_interv.affected_nodes)
                h1 = g.h1_func(input1)
                h2 = g.h2_func(input2)
                a = g.add_func(h1, h2)
                r = g.relu_func(a)
                base_expected = g.root_func(r)

                h2_after = h2.detach().clone()
                h2_after[:, 0] = interv1
                h2_after[:, 2] = interv2

                # a_prime = g.add_func(h1, h2)
                a_prime = g.add_func(h1, h2_after)
                a_prime = a_prime.detach().clone()
                a_prime[:, 1] = torch.ones(batch_size)
                r_prime = g.relu_func(a_prime)
                interv_expected = g.root_func(r_prime)

                h1_before, h1_after = g.intervene_node("h1", batched_interv)
                a_before, a_after = g.intervene_node("add", batched_interv)
                r_before, r_after = g.intervene_node("relu", batched_interv)

                assert torch.allclose(interv_res, interv_expected)
                assert torch.allclose(base_res, base_expected)

                assert torch.allclose(h1, h1_after)
                assert torch.allclose(h1, h1_before)
                assert torch.allclose(a_prime, a_after)
                assert torch.allclose(a, a_before)
                assert torch.allclose(r_prime, r_after)
                assert torch.allclose(r, r_before)

def test_intervene_twice():
    g = TensorArithmeticGraphDim0()
    torch.manual_seed(39)
    input1 = torch.randn(20, 3)
    input2 = torch.randn(20, 3) * 2.
    interv = torch.zeros(20, 2)

    input_dict = {"leaf1": input1, "leaf2": input2}
    interv_dict = {"h2": interv}
    loc_dict = {"h2": LOC[:, 1:]}

    i = Intervention(input_dict, interv_dict, loc_dict, batched=True, batch_dim=0)

    before1, after1 = g.intervene(i)

    i2 = Intervention(input_dict, interv_dict, loc_dict)

    before2, after2 = g.intervene(i2)

    assert torch.allclose(before1, before2)
    assert torch.allclose(after1, after2)