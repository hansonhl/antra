import pytest

import torch
from antra import *
import antra.interchange.batched as batched


def test_pack_interventions_dim0_basic():
    batch_dim = 0
    ivns = [
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])},
                     {"c": LOC[:,1:3]}),
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])},
                     {"c": LOC[:,1:3]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])},
                     {"c": LOC[:,1:3]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])},
                     {"c": LOC[:,1:3]}),
    ]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)
    assert batch_ivn.base["a"].shape == (4,5)
    assert batch_ivn.base["b"].shape == (4,3)
    assert batch_ivn["c"].shape == (4,2)
    assert batch_ivn.location["c"] == LOC[:,1:3]

    assert torch.all(batch_ivn.base["a"] == torch.tensor([[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5],[11,2,3,4,5]]))
    assert torch.all(batch_ivn.base["b"] == torch.tensor([[5,6,7]] * 4))
    assert torch.all(batch_ivn["c"] == torch.tensor([[3,4], [3,4], [2,3], [2,3]]))

def test_pack_interventions_dim0_no_loc():
    batch_dim = 0
    ivns = [
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])}),
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])})
    ]

    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)

    assert batch_ivn.base["a"].shape == (4,5)
    assert batch_ivn.base["b"].shape == (4,3)
    assert batch_ivn["c"].shape == (4,2)
    assert torch.all(batch_ivn.base["a"] == torch.tensor([[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5],[11,2,3,4,5]]))
    assert torch.all(batch_ivn.base["b"] == torch.tensor([[5,6,7]] * 4))
    assert torch.all(batch_ivn["c"] == torch.tensor([[3,4], [3,4], [2,3], [2,3]]))

    assert len(batch_ivn.location) == 0

def test_pack_interventions_dim1_basic():
    batch_dim=1
    ivns = [
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])},
                     {"c": LOC[1:3,:]}),
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([3,4])},
                     {"c": LOC[1:3,:]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])},
                     {"c": LOC[1:3,:]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": torch.tensor([2,3])},
                     {"c": LOC[1:3,:]}),
    ]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)

    assert batch_ivn.base["a"].shape == (5, 4)
    assert batch_ivn.base["b"].shape == (3, 4)
    assert batch_ivn["c"].shape == (2, 4)
    assert batch_ivn.location["c"] == LOC[1:3,:]

    expected_a = torch.tensor([[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5],[11,2,3,4,5]]).T
    expected_b = torch.tensor([[5,6,7]] * 4).T
    expected_c = torch.tensor([[3,4], [3,4], [2,3], [2,3]]).T
    assert torch.all(batch_ivn.base["a"] == expected_a)
    assert torch.all(batch_ivn.base["b"] == expected_b)
    assert torch.all(batch_ivn["c"] == expected_c)

def test_pack_interventions_dim0_multi_loc():
    batch_dim = 0
    ivns = [
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": [torch.tensor([3,4]), torch.tensor([6])]},
                     {"c": [LOC[:,1:3], LOC[:,5]]}),
        Intervention({"a": torch.tensor([1,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": [torch.tensor([3,4]), torch.tensor([6])]},
                     {"c": [LOC[:,1:3], LOC[:,5]]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": [torch.tensor([2,3]), torch.tensor([5])]},
                     {"c": [LOC[:,1:3], LOC[:,5]]}),
        Intervention({"a": torch.tensor([11,2,3,4,5]), "b": torch.tensor([5,6,7])},
                     {"c": [torch.tensor([2,3]), torch.tensor([5])]},
                     {"c": [LOC[:,1:3], LOC[:,5]]}),
    ]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)
    assert batch_ivn.base["a"].shape == (4,5)
    assert batch_ivn.base["b"].shape == (4,3)
    assert batch_ivn["c"][0].shape == (4,2)
    assert batch_ivn["c"][1].shape == (4,1)
    assert batch_ivn.location["c"][0] == LOC[:,1:3]
    assert batch_ivn.location["c"][1] == LOC[:,5]


    expected_a = torch.tensor([[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5],[11,2,3,4,5]])
    expected_b = torch.tensor([[5,6,7]] * 4)
    expected_c_0 = torch.tensor([[3,4], [3,4], [2,3], [2,3]])
    expected_c_1 = torch.tensor([[6], [6], [5], [5]])
    assert torch.all(batch_ivn.base["a"] == expected_a)
    assert torch.all(batch_ivn.base["b"] == expected_b)
    assert torch.all(batch_ivn["c"][0] == expected_c_0)
    assert torch.all(batch_ivn["c"][1] == expected_c_1)

def test_pack_interventions_dim0_multi_dim():
    a_tensor = torch.randn(4,5,6)
    b_tensor = torch.randn(4,4)
    c_tensor = torch.randn(4,2,3)
    batch_dim = 0

    ivns = [
        Intervention({"a": a_tensor[i], "b": b_tensor[i]},
                     {"c": c_tensor[i]},
                     {"c": LOC[:, :2, 4:7]}) for i in range(4)
    ]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)
    assert batch_ivn.base["a"].shape == (4,5,6)
    assert batch_ivn.base["b"].shape == (4,4)
    assert batch_ivn["c"].shape == (4,2,3)
    assert batch_ivn.location["c"] == LOC[:, :2, 4:7]

    assert torch.all(batch_ivn.base["a"] == a_tensor)
    assert torch.all(batch_ivn.base["b"] == b_tensor)
    assert torch.all(batch_ivn["c"] == c_tensor)

def test_pack_interventions_dim1_multi_dim():
    a_tensor = torch.randn(4,5,6)
    b_tensor = torch.randn(4,4)
    c_tensor = torch.randn(4,2,3)
    batch_dim = 1

    ivns = [
        Intervention({"a": a_tensor[i], "b": b_tensor[i]},
                     {"c": c_tensor[i]},
                     {"c": LOC[:2,:, 4:7]}) for i in range(4)
    ]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim
    )
    batch_ivn = Intervention.batched(base_dict, interv_dict, loc_dict, batch_dim=batch_dim)
    assert batch_ivn.base["a"].shape == (5,4,6)
    assert batch_ivn.base["b"].shape == (4,4)
    assert batch_ivn["c"].shape == (2,4,3)
    assert batch_ivn.location["c"] == LOC[:2, :, 4:7]

    expected_a_tensor = torch.transpose(a_tensor, 0, 1)
    expected_b_tensor = torch.transpose(b_tensor, 0, 1)
    expected_c_tensor = torch.transpose(c_tensor, 0, 1)
    assert torch.all(batch_ivn.base["a"] == expected_a_tensor)
    assert torch.all(batch_ivn.base["b"] == expected_b_tensor)
    assert torch.all(batch_ivn["c"] == expected_c_tensor)

def test_pack_interventions_dim0_with_non_batch():
    a_tensor = torch.randn(4,5,6)
    b_tensor = torch.randn(4,4)
    c_tensor = torch.randn(4,2,3)
    batch_dim = 0
    ivns = [
        Intervention({"a": a_tensor[i], "b": b_tensor[i], "x": None, "y": True},
                     {"c": c_tensor[i]},
                     {"c": LOC[:,:2, 4:7]}) for i in range(4)
    ]
    non_batch_leaves = ["x", "y"]
    base_dict, interv_dict, loc_dict = batched.pack_interventions(
        ivns, batch_dim=batch_dim, non_batch_inputs=non_batch_leaves
    )

    base_input = GraphInput.batched(base_dict, non_batch_leaves=non_batch_leaves)
    batch_ivn = Intervention.batched(base_input, interv_dict, loc_dict, batch_dim=batch_dim)

    assert batch_ivn.base["a"].shape == (4,5,6)
    assert batch_ivn.base["b"].shape == (4,4)
    assert batch_ivn["c"].shape == (4,2,3)
    assert batch_ivn.location["c"] == LOC[:, :2, 4:7]

    assert batch_ivn.base["x"] is None
    assert batch_ivn.base["y"] == True
    assert torch.all(batch_ivn.base["a"] == a_tensor)
    assert torch.all(batch_ivn.base["b"] == b_tensor)
    assert torch.all(batch_ivn["c"] == c_tensor)
