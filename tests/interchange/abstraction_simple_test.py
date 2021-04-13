from antra import ComputationGraph, GraphNode, Intervention, Location
from antra.interchange.abstraction import find_abstractions
import numpy as np

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
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate, bool_leaf3, bool_leaf4)
        def bool_root(w,v , z):
            return np.array([float(bool(w[0]) and bool(z[0]) and bool(v[0]))], dtype=np.float64)

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
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate1,bool_leaf3)
        def bool_intermediate2(x,y):
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate2, bool_leaf4)
        def bool_root(w, v):
            return np.array([float(bool(w[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(bool_root)

def test_abstraction_simple():
    high_model= BooleanLogicProgram()
    low_model = BooleanLogicProgram2()
    #for mapping in create_possible_mappings(low_model,high_model, fixed_assignments={x:{x:Location()[:]} for x in ["bool_root", "bool_leaf1",  "bool_leaf2", "bool_leaf3", "bool_leaf4"]}):
    #    print(mapping)
    #    print("done \n\n")

    inputs = []
    for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
        inputs.append(Intervention({"bool_leaf1":x[0],"bool_leaf2":x[1],"bool_leaf3":x[2],"bool_leaf4":x[3], }, dict()))

    print("first input is empty", inputs[0].intervention.is_empty())

    total_high_interventions = []
    for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
        for y in [np.array([0]), np.array([1])]:
            total_high_interventions.append(Intervention({"bool_leaf1":x[0],"bool_leaf2":x[1],"bool_leaf3":x[2],"bool_leaf4":x[3], }, {"bool_intermediate":y}))
    high_model.intervene_node(high_model.root.name,inputs[0])
    low_model.intervene_node(high_model.root.name,inputs[0])

    fixed_assignments =  {x:{x:Location()[:]} for x in ["bool_root", "bool_leaf1",  "bool_leaf2", "bool_leaf3", "bool_leaf4"]}
    results = find_abstractions(low_model, high_model, inputs,total_high_interventions, fixed_assignments, lambda x: x)

    fail_list = []
    for result, mapping in results:
        fail = False
        result, realization_to_inputs = result
        # print("type of result", type(result))
        for interventions in result:
            low_intervention, high_intervention = interventions
            print("mapping", mapping)
            print("low:",low_intervention.intervention.values)
            print("lowbase:",low_intervention.base.values)
            print("high:", high_intervention.intervention.values)
            print("highbase:", high_intervention.base.values)
            print("success:",result[interventions])
            print("\n\n")
            if not result[interventions]:
                fail = True
                if "bool_intermediate1" in mapping["bool_intermediate"]:
                    raise RuntimeError("something wrong happened")
        fail_list.append(fail)
    print(fail_list)
