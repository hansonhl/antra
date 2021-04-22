from antra import *
# from antra.interchange.abstraction_single_old import find_abstractions
from antra.interchange.batched import BatchedInterchange
import torch
from pprint import pprint
from itertools import product

class BooleanLogicProgram(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def leaf1(a):
            return a

        @GraphNode()
        def leaf2(b):
            return b

        @GraphNode()
        def leaf3(c):
            return c

        @GraphNode()
        def leaf4(d):
            return d

        @GraphNode(leaf1,leaf2)
        def node(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(node, leaf3, leaf4)
        def root(w, v , z):
            return w & v & z
            # return np.array([float(bool(w[0]) and bool(z[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(root)

class BooleanLogicProgram2(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def leaf1(a):
            return a

        @GraphNode()
        def leaf2(b):
            return b

        @GraphNode()
        def leaf3(c):
            return c

        @GraphNode()
        def leaf4(d):
            return d

        @GraphNode(leaf1,leaf2)
        def node1(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(node1,leaf3)
        def node2(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(node2, leaf4)
        def root(w, v):
            return w & v
            # return np.array([float(bool(w[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(root)

def test_abstraction_simple():
    high_model= BooleanLogicProgram()
    low_model = BooleanLogicProgram2()
    #for mapping in create_possible_mappings(low_model,high_model, fixed_assignments={x:{x:Location()[:]} for x in ["root", "leaf1",  "leaf2", "leaf3", "leaf4"]}):
    #    print(mapping)
    #    print("done \n\n")

    inputs = [
        GraphInput({
            "leaf1": torch.tensor([a]),
            "leaf2": torch.tensor([b]),
            "leaf3": torch.tensor([c]),
            "leaf4": torch.tensor([d])
        })
        for (a, b, c, d) in product((True, False), repeat=4)
    ]

    total_high_interventions = [
        Intervention({
            "leaf1": torch.tensor([a]),
            "leaf2": torch.tensor([b]),
            "leaf3": torch.tensor([c]),
            "leaf4": torch.tensor([d])
        }, {
            "node": torch.tensor([y])
        })
        for (a, b, c, d, y) in product((True, False), repeat=5)
    ]

    # total_high_interventions = []
    # for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    #     for y in [np.array([0]), np.array([1])]:
    #         total_high_interventions.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2],"leaf4":x[3], }, {"node":y}))
    # high_model.intervene_node(high_model.root.name,inputs[0])
    # low_model.intervene_node(high_model.root.name,inputs[0])

    fixed_node_mapping =  {x: {x: None} for x in ["root", "leaf1",  "leaf2", "leaf3", "leaf4"]}

    ca = BatchedInterchange(
        low_model=low_model,
        high_model=high_model,
        low_inputs=inputs,
        high_inputs=inputs,
        high_interventions=total_high_interventions,
        fixed_node_mapping=fixed_node_mapping,
        result_format="verbose",
        batch_size=12,
    )
    assert len(ca.high_intervention_range) == 1
    assert ca.high_intervention_range["node"] == {(False,), (True,)}

    assert len(ca.mappings) == 2

    # visual inspection
    for k, v in ca.high_to_low_input_keys.items():
        assert k == v
    print("mappings")
    pprint(ca.mappings)
    print("high intervention range")
    pprint(ca.high_intervention_range)

    mapping1 = ca.mappings[0] if "node1" in ca.mappings[0]["node"] else ca.mappings[1]
    mapping2 = ca.mappings[0] if "node2" in ca.mappings[0]["node"] else ca.mappings[1]

    # test mapping 1
    print("========== testing mapping 1 ==========")
    pprint(mapping1)

    result1 = ca.test_mapping(mapping1)

    # return
    success1 = True
    for ivn_keys in result1:
        low_ivn_key, high_ivn_key = ivn_keys
        low_ivn = ca.low_keys_to_interventions[low_ivn_key]
        high_ivn = ca.high_keys_to_interventions[high_ivn_key]

        leaf3 = low_ivn.base["leaf3"]
        leaf4 = low_ivn.base["leaf4"]
        node = high_ivn.intervention["node"]
        high_res = node & leaf3 & leaf4

        node1 = low_ivn.intervention["node1"]
        assert node1 == node
        low_res = node1 & leaf3 & leaf4

        assert result1[ivn_keys] == (high_res == low_res)

        # print("---- High ivn")
        # print(high_ivn)
        # print("---- Low ivn")
        # print(low_ivn)
        # print(f"RESULT: {result1[ivn_keys]}\n")

        if not result1[ivn_keys]: success1 = False

    assert success1


    print("========== testing mapping 2 ==========")
    pprint(mapping2)
    result2 = ca.test_mapping(mapping2)
    success2 = True

    for ivn_keys in result2:
        if not result2[ivn_keys]: success2 = False
        low_ivn_key, high_ivn_key = ivn_keys
        low_ivn = ca.low_keys_to_interventions[low_ivn_key]
        high_ivn = ca.high_keys_to_interventions[high_ivn_key]

        leaf3 = low_ivn.base["leaf3"]
        leaf4 = low_ivn.base["leaf4"]
        node = high_ivn.intervention["node"]
        high_res = node & leaf3 & leaf4

        node2 = low_ivn.intervention["node2"]
        low_res = node2 & leaf4

        assert result2[ivn_keys] == (high_res == low_res)


    assert not success2


    # mapping = ca.mappings[0]
    #
    # results = ca.find_abstractions()
    # # result = ca.test_mapping(mapping)
    # # print(f"Got {len(result)} results")
    #
    #
    # success_list = []
    # for result, mapping in results:
    #     #     fail = False
    #     # result, realization_to_inputs = result
    #     # print("type of result", type(result))
    #     success = True
    #     for interventions in result:
    #         low_intervention, high_intervention = interventions
    #         # print("mapping", mapping)
    #         # print("low intervention key")
    #         # pprint(low_intervention)
    #         # print("high intervention key")
    #         # pprint(high_intervention)
    #         # print("low:",low_intervention.intervention.values)
    #         # print("lowbase:",low_intervention.base.values)
    #         # print("high:", high_intervention.intervention.values)
    #         # print("highbase:", high_intervention.base.values)
    #         # print("success:",result[interventions])
    #         # print("\n\n")
    #         if not result[interventions]:
    #             success = False
    #             if "node1" in mapping["node"]:
    #                 raise RuntimeError("something wrong happened")
    #     success_list.append(success)
    # print(success_list)
    # print("Success?", success)
