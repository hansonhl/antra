import pytest
import torch
from antra import GraphInput, Intervention, LOC

from .utils import setup_intervention

@pytest.fixture()
def base_input():
    d = {str(x): torch.randn(10) for x in range(3)}
    return GraphInput(d)


@pytest.fixture()
def interv1():
    return {"node": torch.randn(10)}


@pytest.fixture()
def interv1_loc():
    return {"node[5 :10]": torch.randn(5)}


@pytest.fixture()
def loc1():
    return {"node": LOC[5:10]}


@pytest.fixture()
def interv2():
    return {"node2": torch.randn(10)}


@pytest.fixture()
def interv2_loc():
    return {"node2[0,...,:]": torch.randn(10)}


@pytest.fixture()
def loc2():
    return {"node2": LOC[0,...,:]}


def test_init_base_only(base_input):
    i = Intervention(base_input)
    assert isinstance(i.location, dict) and len(i.location) == 0
    assert isinstance(i.intervention, GraphInput) and len(i.intervention) == 0


def test_init_base_and_interv(base_input, interv1):
    i = Intervention(base_input, intervention=interv1)
    assert i.location == {}


def test_init_with_loc1(base_input, interv1_loc):
    interv_dict = {"node[5 :10]": torch.randn(5)}
    i = Intervention(base_input, intervention=interv_dict)
    assert "node" in i.intervention
    assert i.location["node"] == LOC[5:10]


def test_init_with_loc2(base_input, interv1, loc1):
    interv_dict = {"node": torch.randn(10)}
    loc_dict = {"node": LOC[5:10]}
    i = Intervention(base_input, intervention=interv_dict, location=loc_dict)
    assert i.location["node"] == LOC[5:10]


# def test_init_with_loc3(base_input, interv1_loc, loc1):
#     interv_dict = {"node[5 :10]": torch.randn(5)}
#     loc_dict = {"node": LOC[5:10]}
#     i = Intervention(base_input, intervention=interv_dict, location=loc_dict)
#     assert i.location["node"] == LOC[5:10]


def test_init_with_loc4(base_input, interv1_loc, interv2, loc2):
    interv1_loc.update(interv2)
    interv_dict = {"node[5 :10]": torch.randn(5), "node2": torch.randn(10)}
    loc_dict = {"node2": LOC[0,...,:]}
    i = Intervention(base_input, intervention=interv_dict, location=loc_dict)
    assert i.location["node2"] == LOC[0, ..., :]


def test_set_intervention(base_input):
    i = Intervention(base_input)
    i.set_intervention("node", torch.randn(10))
    assert len(i.intervention["node"]) == 10
    interv_input = i.intervention

    i.set_intervention("node2", torch.randn(10))
    interv_input2 = i.intervention

    # i.intervention should be an immutable GraphInput object; setting it should
    # produce a new instance
    assert interv_input is not interv_input2
    assert len(i.intervention["node"]) == 10 and len(i.intervention["node2"]) == 10


def test_loc(base_input, interv1_loc):
    i = Intervention(base_input, intervention=interv1_loc)
    loc = i.location["node"]
    x = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])[loc]
    y = torch.tensor([10, 12, 14, 16, 18])
    assert torch.all(torch.eq(x, y))


# def test_device(base_input, interv1_loc):
#     device = torch.device("cuda")
#     base_input = base_input.to(device)
#     interv1_loc = {k: v.to(device) for k, v in interv1_loc.items()}
#     i = Intervention(base_input, intervention=interv1_loc)
#
#     assert all(t.is_cuda for t in i.base.values.values())
#     assert all(v.is_cuda for v in i.intervention.values.values())

def test_double_set():
    i = Intervention({"leaf1": torch.tensor([1,2,3])})
    i.set_intervention("node", torch.tensor([2]))
    i.set_location("node", LOC[1])
    i.set_intervention("node", torch.tensor([3]))
    i.set_location("node", LOC[2])

    assert i.intervention["node"] == torch.tensor([3])
    assert i.location["node"] == LOC[2]

def test_one_node_multi_interv(setup_intervention):
    input_dict = {"leaf1": torch.tensor([-2., 3., 1., ])}
    interv_dict = {"h1": [torch.tensor([-6.]), torch.tensor([10.]), torch.tensor([1.])]}
    loc_dict = {"h1": [0, 2, 1]}

    i = setup_intervention(input_dict, interv_dict, loc_dict)

    assert len(i.intervention["h1"]) == 3
    assert i.intervention["h1"][0] == torch.tensor([-6.])
    assert i.intervention["h1"][1] == torch.tensor([10.])
    assert i.intervention["h1"][2] == torch.tensor([1.])
    assert i.location["h1"] == [0, 2, 1]

def test_multi_node_multi_interv(setup_intervention):
    input_dict = {"leaf1": torch.tensor([-2., 3., 1.])}
    interv_dict = {"h1": [torch.tensor([-6.]), torch.tensor([10.]), torch.tensor([1.])],
                   "node": torch.tensor([1., 2., 3.]),
                   "h2": [torch.tensor([-3.]), torch.tensor([-3., -2.])]}
    loc_dict = {"h1": [0, 2, 1], "node": LOC[1:4], "h2": [1, LOC[3:5]]}

    i = setup_intervention(input_dict, interv_dict, loc_dict)

    assert len(i.intervention["h1"]) == 3
    assert i.intervention["h1"][0] == torch.tensor([-6.])
    assert i.intervention["h1"][1] == torch.tensor([10.])
    assert i.intervention["h1"][2] == torch.tensor([1.])
    assert i.location["h1"] == [0, 2, 1]

    assert len(i.intervention["h2"]) == 2
    assert i.intervention["h2"][0] == torch.tensor([-3.])
    assert torch.allclose(i.intervention["h2"][1], torch.tensor([-3., -2.]))
    assert i.location["h2"] == [1, LOC[3:5]]

    assert torch.allclose(i.intervention["node"], torch.tensor([1., 2., 3.]))
    assert i.location["node"] == LOC[1:4]
