import itertools
import pytest
from antra import *

##### Try various combinations of setting up an intervention

interv_construction_types = [("dict", "GraphInput"),
                             ("dict", "GraphInput", "set"),
                             ("dict", "interv_str", "set")]
invalid_construction_tuples = {
    ("dict", "set", "dict"),
    ("GraphInput", "set", "dict")
}
params = [t for t in itertools.product(*interv_construction_types) if t not in invalid_construction_tuples]
# params = [("dict", "set", "dict")]
idfn = lambda t: "/".join(t)
@pytest.fixture(params=params, ids=idfn)
def setup_intervention(request):
    return setup_intervention_func_for_fixture(request)



def setup_intervention_func_for_fixture(request):
    base_method, interv_method, loc_method = request.param

    def _setup_intervention(input_dict, interv_dict, loc_dict=None, batched=False, batch_dim=0):
        if base_method == "dict":
            base_input = input_dict
        else:
            base_input = GraphInput(input_dict, batched=batched, batch_dim=batch_dim)

        interv_dict = interv_dict.copy()
        if loc_dict and loc_method == "interv_str":
            for interv_node_name, locs in loc_dict.items():
                if not isinstance(locs, list):
                    loc_str = Location.loc_to_str(locs, add_brackets=True)
                    new_interv_node_name = interv_node_name + loc_str
                    interv_dict[new_interv_node_name] = interv_dict[interv_node_name]
                    del interv_dict[interv_node_name]
                else:
                    interv_values = interv_dict[interv_node_name]
                    for loc, interv_value in zip(locs, interv_values):
                        loc_str = Location.loc_to_str(loc, add_brackets=True)
                        new_interv_node_name = interv_node_name + loc_str
                        interv_dict[new_interv_node_name] = interv_value
                    del interv_dict[interv_node_name]

        if interv_method == "dict":
            interv = interv_dict
        elif interv_method == "GraphInput":
            interv = GraphInput(interv_dict, batched=batched, batch_dim=batch_dim)
        else:
            interv = None

        if loc_method == "dict":
            locs = loc_dict
        else:
            locs = None

        intervention = Intervention(base_input, interv, locs, batched=batched, batch_dim=batch_dim)

        if interv_method == "set":
            for interv_node_name, interv_value in interv_dict.items():
                # print(f"set_intervention({interv_node_name, interv_value})")
                intervention.set_intervention(interv_node_name, interv_value)
        if loc_dict and loc_method == "set":
            for interv_node_name, loc in loc_dict.items():
                # print(f"set_loc({interv_node_name, loc})")
                intervention.set_location(interv_node_name, loc)

        return intervention

    return _setup_intervention
