from antra import ComputationGraph, GraphNode, Location
from antra.interchange.mapping import create_possible_mappings


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
            return [[int(bool(x[0]) and bool(y[0]))]]

        @GraphNode(bool_intermediate, bool_leaf3, bool_leaf4)
        def bool_root(w,v , z):
            return [[int(bool(w[0]) and bool(z[0])) and bool(v[0])]]

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
            return [[int(bool(x[0]) and bool(y[0]))]]

        @GraphNode(bool_intermediate1,bool_leaf3)
        def bool_intermediate2(x,y):
            return [[int(bool(x[0]) and bool(y[0]))]]

        @GraphNode(bool_intermediate2, bool_leaf4)
        def bool_root(w,v , z):
            return [[int(bool(w[0]) and bool(z[0])) and bool(v[0])]]

        super().__init__(bool_root)

# model1 = BooleanLogicProgram()
# model2 = BooleanLogicProgram2()
# for mapping in create_possible_mappings(model2,model1, fixed_assignments={x:{x:Location()[:]} for x in ["bool_root", "bool_leaf1",  "bool_leaf2", "bool_leaf3", "bool_leaf4"]} ):
#     print(mapping)
#     print("\n\n")
