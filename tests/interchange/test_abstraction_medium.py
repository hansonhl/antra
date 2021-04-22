from itertools import product
from pprint import pprint

from antra import *
from antra.interchange import BatchedInterchange
from antra.location import location_to_str, reduce_dim

import torch
import torch.nn.functional as F
# from causal_abstraction.abstraction import find_abstractions
# import numpy as np
# from causal_abstraction.clique_analysis import construct_graph, find_cliques

class BooleanLogicProgram(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")

        @GraphNode(leaf1,leaf2)
        def node(x,y):
            return x & y

        @GraphNode(node, leaf3)
        def root(w,v ):
            return w & v

        super().__init__(root)

class NeuralNetwork(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")
        self.W = torch.tensor([[0.5, 0.5, 0.], [0.5, 0.5, 0.], [1., -1., 1.]])
        self.B = torch.tensor([-1., -1., 0.])
        self.w = torch.tensor([1.,1.,1.])
        self.b = 1.5

        @GraphNode(leaf1,leaf2, leaf3)
        def hidden1(x,y,z):
            a = torch.stack([x, y, z], dim=-1)
            h = a.matmul(self.W) + self.B
            h = F.relu(h)
            return h

        @GraphNode(hidden1)
        def hidden2(x):
            return x.matmul(self.w) - self.b

        @GraphNode(hidden2)
        def root(x):
            return x > 0 # bool tensor

        super().__init__(root)

# def verify_intervention(mapping, low_intervention, high_intervention, result):
#     intermediate_high = bool(high_intervention.base.values["leaf1"][0]) and bool(high_intervention.base.values["leaf2"][0])
#     if "bool_intermediate" in high_intervention.intervention.values:
#         intermediate_high = bool(high_intervention.intervention.values["bool_intermediate"][0])
#     output_high = intermediate_high and bool(high_intervention.base.values["leaf3"][0])
#     a = np.array([low_intervention.base.values["leaf1"][0],low_intervention.base.values["leaf2"][0], low_intervention.base.values["leaf3"][0]])
#     print(a)
#     h = np.matmul(a,np.array([[0.5,0.5,0],[0.5,0.5,0],[1,-1,1]])) + np.array([-1,-1,0])
#     print(h)
#     h = np.maximum(h, 0)
#     print(h)
#     if "hidden1" in low_intervention.intervention.values:
#         h[mapping["bool_intermediate"]["hidden1"]] = low_intervention.intervention.values["hidden1"]
#     y = np.matmul(h, np.transpose(np.array([1,1,1]))) - 1.5
#     if "hidden2" in low_intervention.intervention.values:
#         y = low_intervention.intervention.values["hidden2"]
#     output_low = y > 0
#     if (output_low == output_high) == result[0]:
#         return
#     print(afwoeijfaoeifj)

def verify_mapping(ca, mapping, result, low_inputs, low_model):
    print("mapping", mapping)
    print("results", len(result))
    # if len(result.keys()) not in [32, 40]:
    #     print(len(result.keys()))
    #     print(fawefawe)
    for key in mapping["node"]:
        low_node = key
    low_locs = mapping["node"][low_node]
    if not isinstance(low_locs, list):
        low_locs = [low_locs]
    low_indices = [reduce_dim(l, 0) for l in low_locs]
    realizations = []

    for low_input in low_inputs:
        # low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in input.base.values}, dict())
        low_val = low_model.compute_node(low_node, low_input)
        rzn = [low_val[low_loc] for low_loc in low_indices]
        realizations.append(rzn)
    # print(f"{realizations=}")

    pairs_to_verify = []
    for low_input in low_inputs:
        for realization in realizations:
            pairs_to_verify.append((low_input, realization))
    # convert = {0:-1, 1:1}
    success = set()

    for keys in result:
        low_ivn_key, high_ivn_key = keys
        low_ivn = ca.low_keys_to_interventions[low_ivn_key]
        low_ivn_value = low_ivn.intervention.values[low_node]
        if not isinstance(low_ivn_value, list):
            low_ivn_value = [low_ivn_value]

        for i, (low_input, realization) in enumerate(pairs_to_verify):
            assert isinstance(low_ivn_value, list)
            assert isinstance(realization, list)

            ok = all(torch.allclose(_val, _rzn) for _val, _rzn in zip(low_ivn_value, realization))
            ok &= all(low_ivn.base[f"leaf{k}"] == low_input[f"leaf{k}"] for k in range(1,4))
            if ok: success.add(i)

                # if int(low_ivn.base["leaf1"]) == convert[int(low_input["leaf1"])]:
                #     if int(low_ivn.base["leaf2"]) == convert[int(low_input["leaf2"])]:
                #         if int(low_ivn.base["leaf3"]) == convert[int(low_input["leaf3"])]:
                #             success.add(i)

    unrealized_ivns = set(range(len(pairs_to_verify))) - success
    assert len(unrealized_ivns) == 0
    # print(f"{len(unrealized_ivns)} / {len(pairs_to_verify)} unrealized interventions")

    # for i in sorted(unrealized_ivns):
    #     low_input, rzn = pairs_to_verify[i]
    #     print(f"{low_input=}, {rzn=}")

    # for i in range(len(pairs_to_verify)):
    #     if i not in success:
    #         print(fawefawe)

def test_find_abstraction():
    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork()

    for (a, b, c) in product((-1., 1.), repeat=3):
        li = GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        })
        node = (a == 1) and (b == 1)
        expected = (a==1) and (b == 1) and (c == 1)
        hidden1_val = low_model.compute_node("hidden1", li)
        hidden2_val = low_model.compute_node("hidden2", li)
        print(f"a & b = {node}, hidden1 = {hidden1_val}, hidden2 = {hidden2_val}")
        assert expected == low_model.compute(li)



def test_mapping_generation():
    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork()
    low_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((-1., 1.), repeat=3)
    ]

    high_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((False, True), repeat=3)
    ]

    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c),
        }, {
            "node": torch.tensor(y)
        })
        for (a, b, c, y) in product((True, False), repeat=4)
    ]

    fixed_node_mapping =  {x: {x: None} for x in ["root", "leaf1",  "leaf2", "leaf3"]}
    low_nodes_to_indices = {
        "hidden2": [None],
        "hidden1": [LOC[:,0] , LOC[:, 1], LOC[:, 2], LOC[:,:1], LOC[:,1:], None]
    }

    ca = BatchedInterchange(
        low_model=low_model,
        high_model=high_model,
        low_inputs=low_inputs,
        high_inputs=high_inputs,
        high_interventions=high_ivns,
        low_nodes_to_indices=low_nodes_to_indices,
        fixed_node_mapping=fixed_node_mapping,
        result_format="verbose",
        batch_size=12,
    )

    print(ca.mappings)
    assert len(ca.mappings) == 7
    for low_node, low_locs in low_nodes_to_indices.items():
        for low_loc in low_locs:
            ok = False
            for m in ca.mappings:
                if low_node not in m["node"]: continue
                if m["node"][low_node] == low_loc:
                    ok = True
            assert ok

def test_abstraction_medium():
    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork()
    low_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((-1., 1.), repeat=3)
    ]

    high_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((False, True), repeat=3)
    ]

    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c),
        }, {
            "node": torch.tensor(y)
        })
        for (a, b, c, y) in product((True, False), repeat=4)
    ]

    fixed_node_mapping =  {x: {x: None} for x in ["root", "leaf1",  "leaf2", "leaf3"]}
    low_nodes_to_indices = {
        "hidden2": [None],
        "hidden1": [LOC[:,0] , LOC[:, 1], LOC[:, 2], LOC[:,:2], LOC[:,1:], LOC[:, :]]
    }

    # TODO: disjoint location [LOC[:,0], LOC[:,2]]

    ca = BatchedInterchange(
        low_model=low_model,
        high_model=high_model,
        low_inputs=low_inputs,
        high_inputs=high_inputs,
        high_interventions=high_ivns,
        low_nodes_to_indices=low_nodes_to_indices,
        fixed_node_mapping=fixed_node_mapping,
        result_format="verbose",
        batch_size=12,
    )

    find_abstr_res = ca.find_abstractions()


    success_list = []
    for result, mapping in find_abstr_res:
        low_node = list(mapping["node"].keys())[0]
        low_loc = mapping["node"][low_node]
        red_low_loc = reduce_dim(low_loc, 0)
        if low_loc is None: continue
        print(f"Low node and loc: {low_node}{location_to_str(low_loc,add_brackets=True)}")
        print(f"Reduced loc: {red_low_loc}")

        success = True
        verify_mapping(ca, mapping, result, low_inputs, low_model)
        # G, causal_edges = construct_graph(low_model,high_model, mapping, result, realizations_to_inputs, "node", "root")
        # cliques = find_cliques(G, causal_edges, 5)

        # print("cliques:", cliques)
        # x = input()

        for keys in result:
            # low_intervention, high_intervention = interventions
            low_ivn_key, high_ivn_key = keys
            low_ivn = ca.low_keys_to_interventions[low_ivn_key]
            high_ivn = ca.high_keys_to_interventions[high_ivn_key]

            _, low_res = low_model.intervene(low_ivn)
            _, high_res = high_model.intervene(high_ivn)

            x, y, z = tuple(low_ivn.base[f"leaf{i}"] for i in range(1,4))
            # print(f"{x=}")
            a = torch.stack([x, y, z], dim=-1)
            h = a.matmul(low_model.W) + low_model.B
            h = F.relu(h)

            assert torch.allclose(h, low_model.compute_node('hidden1', low_ivn.base))
            if low_node == "hidden1":
                ivn_value = low_ivn.intervention[low_node]
                h[red_low_loc] = ivn_value

            h2 = h.matmul(low_model.w) - low_model.b
            if low_node == "hidden2":
                ivn_value = low_ivn.intervention[low_node]
                h2 = ivn_value

            expected_low_res = h2 > 0

            n = high_ivn.base["leaf1"] & high_ivn.base["leaf2"]
            if "node" in high_ivn.intervention:
                n = high_ivn.intervention["node"]
            expected_high_res = n & high_ivn.base["leaf3"]

            assert expected_low_res == low_res
            assert expected_high_res == high_res
            assert (expected_low_res == expected_high_res) == result[keys]

            # print(f"{expected_low_res=}")
            # print(f"{expected_high_res=}")
            #
            # print("low_res", low_res)
            # print("high_res", high_res)
            # print("low:",low_ivn.intervention.values)
            # print("lowbase:", low_ivn.base.values)
            # print("high:", high_ivn.intervention.values)
            # print("highbase:", high_ivn.base.values)
            # print("success:", result[keys])
            #
            # print("\n\n")
            # verify_intervention(mapping,low_intervention, high_intervention, result[keys])

            if not result[keys]:
                success = False

        success_list.append((success,mapping))

    pprint(success_list)


def test_abstraction_medium_multi_loc():
    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork()
    low_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((-1., 1.), repeat=3)
    ]

    high_inputs = [
        GraphInput({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c)
        }) for (a, b, c) in product((False, True), repeat=3)
    ]

    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(a),
            "leaf2": torch.tensor(b),
            "leaf3": torch.tensor(c),
        }, {
            "node": torch.tensor(y)
        })
        for (a, b, c, y) in product((True, False), repeat=4)
    ]

    fixed_node_mapping =  {x: {x: None} for x in ["root", "leaf1",  "leaf2", "leaf3"]}
    low_nodes_to_indices = {
        "hidden2": [None],
        "hidden1": [[LOC[:,0], LOC[:,2]]]
    }

    # TODO: disjoint location [LOC[:,0], LOC[:,2]]

    ca = BatchedInterchange(
        low_model=low_model,
        high_model=high_model,
        low_inputs=low_inputs,
        high_inputs=high_inputs,
        high_interventions=high_ivns,
        low_nodes_to_indices=low_nodes_to_indices,
        fixed_node_mapping=fixed_node_mapping,
        result_format="verbose",
        batch_size=12,
    )

    # pprint(ca.mappings)

    whole_vec_mapping = ca.mappings[0] if "hidden1" in ca.mappings[0]["node"] else ca.mappings[1]
    low_locs = whole_vec_mapping["node"]["hidden1"]
    red_low_locs = [location.reduce_dim(l, 0) for l in low_locs]
    result = ca.test_mapping(whole_vec_mapping)

    verify_mapping(ca, whole_vec_mapping, result, low_inputs, low_model)

    for keys in result:
        # low_intervention, high_intervention = interventions
        low_ivn_key, high_ivn_key = keys
        low_ivn = ca.low_keys_to_interventions[low_ivn_key]
        high_ivn = ca.high_keys_to_interventions[high_ivn_key]

        _, low_res = low_model.intervene(low_ivn)
        _, high_res = high_model.intervene(high_ivn)

        x, y, z = tuple(low_ivn.base[f"leaf{i}"] for i in range(1,4))
        # print(f"{x=}")
        a = torch.stack([x, y, z], dim=-1)
        h = a.matmul(low_model.W) + low_model.B
        h = F.relu(h)

        assert torch.allclose(h, low_model.compute_node('hidden1', low_ivn.base))

        # do intervention
        ivn_values = low_ivn.intervention["hidden1"]
        assert isinstance(ivn_values, list)
        assert len(ivn_values) == 2
        for loc, v in zip(red_low_locs, ivn_values):
            h[loc] = v

        h2 = h.matmul(low_model.w) - low_model.b

        expected_low_res = h2 > 0

        n = high_ivn.base["leaf1"] & high_ivn.base["leaf2"]
        if "node" in high_ivn.intervention:
            n = high_ivn.intervention["node"]
        expected_high_res = n & high_ivn.base["leaf3"]

        assert expected_low_res == low_res
        assert expected_high_res == high_res
        assert (expected_low_res == expected_high_res) == result[keys]

    whole_vec_mapping = ca.mappings[0] if "hidden2" in ca.mappings[0]["node"] else ca.mappings[1]
    result = ca.test_mapping(whole_vec_mapping)

    verify_mapping(ca, whole_vec_mapping, result, low_inputs, low_model)

    for keys in result:
        # low_intervention, high_intervention = interventions
        low_ivn_key, high_ivn_key = keys
        low_ivn = ca.low_keys_to_interventions[low_ivn_key]
        high_ivn = ca.high_keys_to_interventions[high_ivn_key]

        _, low_res = low_model.intervene(low_ivn)
        _, high_res = high_model.intervene(high_ivn)

        x, y, z = tuple(low_ivn.base[f"leaf{i}"] for i in range(1,4))
        # print(f"{x=}")
        a = torch.stack([x, y, z], dim=-1)
        h = a.matmul(low_model.W) + low_model.B
        h = F.relu(h)

        assert torch.allclose(h, low_model.compute_node('hidden1', low_ivn.base))

        # do intervention
        h2 = h.matmul(low_model.w) - low_model.b
        ivn_values = low_ivn.intervention["hidden2"]
        h2 = ivn_values

        expected_low_res = h2 > 0

        n = high_ivn.base["leaf1"] & high_ivn.base["leaf2"]
        if "node" in high_ivn.intervention:
            n = high_ivn.intervention["node"]
        expected_high_res = n & high_ivn.base["leaf3"]

        assert expected_low_res == low_res
        assert expected_high_res == high_res
        assert (expected_low_res == expected_high_res) == result[keys]

    # success_list = []
    # for result, mapping in find_abstr_res:
    #     low_node = list(mapping["node"].keys())[0]
    #     low_loc = mapping["node"][low_node]
    #     red_low_loc = reduce_dim(low_loc, 0)
    #     if low_loc is None: continue
    #     print(f"Low node and loc: {low_node}{location_to_str(low_loc,add_brackets=True)}")
    #     print(f"Reduced loc: {red_low_loc}")
    #
    #     success = True
    #     verify_mapping(ca, mapping, result, low_inputs, low_model)
    #     # G, causal_edges = construct_graph(low_model,high_model, mapping, result, realizations_to_inputs, "node", "root")
    #     # cliques = find_cliques(G, causal_edges, 5)
    #
    #     # print("cliques:", cliques)
    #     # x = input()
    #

    #
    #         # print(f"{expected_low_res=}")
    #         # print(f"{expected_high_res=}")
    #         #
    #         # print("low_res", low_res)
    #         # print("high_res", high_res)
    #         # print("low:",low_ivn.intervention.values)
    #         # print("lowbase:", low_ivn.base.values)
    #         # print("high:", high_ivn.intervention.values)
    #         # print("highbase:", high_ivn.base.values)
    #         # print("success:", result[keys])
    #         #
    #         # print("\n\n")
    #         # verify_intervention(mapping,low_intervention, high_intervention, result[keys])
    #
    #         if not result[keys]:
    #             success = False
    #
    #     success_list.append((success,mapping))
    #
    # pprint(success_list)

#for mapping in create_possible_mappings(low_model,high_model, fixed_assignments={x:{x:Location()[:]} for x in ["bool_root", "leaf1",  "leaf2", "leaf3", "leaf4"]}):
#    print(mapping)
#    print("done \n\n")
