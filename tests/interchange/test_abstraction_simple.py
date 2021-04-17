from antra import *
# from antra.interchange.abstraction_single_old import find_abstractions
from antra.interchange.abstraction import CausalAbstraction
import torch
from pprint import pprint
from itertools import product

class BooleanLogicProgram(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def bool_leaf1(a):
            return a

        @GraphNode()
        def bool_leaf2(b):
            return b

        @GraphNode()
        def bool_leaf3(c):
            return c

        @GraphNode()
        def bool_leaf4(d):
            return d

        @GraphNode(bool_leaf1,bool_leaf2)
        def bool_intermediate(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate, bool_leaf3, bool_leaf4)
        def bool_root(w, v , z):
            return w & v & z
            # return np.array([float(bool(w[0]) and bool(z[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(bool_root)

class BooleanLogicProgram2(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def bool_leaf1(a):
            return a

        @GraphNode()
        def bool_leaf2(b):
            return b

        @GraphNode()
        def bool_leaf3(c):
            return c

        @GraphNode()
        def bool_leaf4(d):
            return d

        @GraphNode(bool_leaf1,bool_leaf2)
        def bool_intermediate1(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate1,bool_leaf3)
        def bool_intermediate2(x,y):
            return x & y
            # return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate2, bool_leaf4)
        def bool_root(w, v):
            return w & v
            # return np.array([float(bool(w[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(bool_root)

def test_abstraction_simple():
    high_model= BooleanLogicProgram()
    low_model = BooleanLogicProgram2()
    #for mapping in create_possible_mappings(low_model,high_model, fixed_assignments={x:{x:Location()[:]} for x in ["bool_root", "bool_leaf1",  "bool_leaf2", "bool_leaf3", "bool_leaf4"]}):
    #    print(mapping)
    #    print("done \n\n")

    inputs = [
        GraphInput({
            "bool_leaf1": torch.tensor([a]),
            "bool_leaf2": torch.tensor([b]),
            "bool_leaf3": torch.tensor([c]),
            "bool_leaf4": torch.tensor([d])
        })
        for (a, b, c, d) in product((True, False), repeat=4)
    ]

    total_high_interventions = [
        Intervention({
            "bool_leaf1": torch.tensor([a]),
            "bool_leaf2": torch.tensor([b]),
            "bool_leaf3": torch.tensor([c]),
            "bool_leaf4": torch.tensor([d])
        }, {
            "bool_intermediate": torch.tensor([y])
        })
        for (a, b, c, d, y) in product((True, False), repeat=5)
    ]

    # total_high_interventions = []
    # for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    #     for y in [np.array([0]), np.array([1])]:
    #         total_high_interventions.append(Intervention({"bool_leaf1":x[0],"bool_leaf2":x[1],"bool_leaf3":x[2],"bool_leaf4":x[3], }, {"bool_intermediate":y}))
    # high_model.intervene_node(high_model.root.name,inputs[0])
    # low_model.intervene_node(high_model.root.name,inputs[0])

    fixed_node_mapping =  {x: {x: None} for x in ["bool_root", "bool_leaf1",  "bool_leaf2", "bool_leaf3", "bool_leaf4"]}

    ca = CausalAbstraction(
        low_model=low_model,
        high_model=high_model,
        low_inputs=inputs,
        high_inputs=inputs,
        high_interventions=total_high_interventions,
        fixed_node_mapping=fixed_node_mapping,
        result_format="verbose",
        batch_size=2,
    )
    assert len(ca.high_intervention_range) == 1
    assert ca.high_intervention_range["bool_intermediate"] == {(False,), (True,)}

    assert len(ca.mappings) == 2

    # visual inspection
    for k, v in ca.high_to_low_input_keys.items():
        assert k == v
    print("mappings")
    pprint(ca.mappings)
    print("high intervention range")
    pprint(ca.high_intervention_range)

    results = ca.find_abstractions()

    fail_list = []
    for result, mapping in results:
        fail = False
        # result, realization_to_inputs = result
        # print("type of result", type(result))
        for interventions in result:
            low_intervention, high_intervention = interventions
            # print("mapping", mapping)
            print("low intervention key")
            pprint(low_intervention)
            print("high intervention key")
            pprint(high_intervention)
            # print("low:",low_intervention.intervention.values)
            # print("lowbase:",low_intervention.base.values)
            # print("high:", high_intervention.intervention.values)
            # print("highbase:", high_intervention.base.values)
            print("success:",result[interventions])
            print("\n\n")
            if not result[interventions]:
                fail = True
                if "bool_intermediate1" in mapping["bool_intermediate"]:
                    raise RuntimeError("something wrong happened")
        fail_list.append(fail)
    print(fail_list)
