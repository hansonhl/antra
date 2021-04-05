# This script is the same as `causal_abstraction/interchange.py`.
# It is duplicated here to support old pickled experiment result files.

import torch

import antra
from antra.utils import serialize
from antra.interchange.mapping import create_possible_mappings

from torch.utils.data import IterableDataset, DataLoader, Subset
from itertools import product

class InterchangeDataset(IterableDataset):
    def __init__(self, mapping, high_interventions, high_inputs):
        super(InterchangeDataset, self).__init__()
        self.low_inputs = []
        self.low_outputs = []
        self.high_interventions = high_interventions
        self.high_outputs = []
        self.realizations = dict() # Maps a high variable and value (V_H, v_H) to vectors of low-values
        self.curr_realizations = dict()# Maps a high variable and value (V_H, v_H) to vectors of low-values, but just the last round
        self.new_realizations = None #Maps a high variable and value (V_H, v_H) to vectors of low-values, but just the new round
        self.mapping = dict()
        self.populate_dataset()


    @property
    def num_examples(self):
        return len(self.low_outputs)

    def update_realizations(self, new_realizations):
        self.realizations = merge_dicts(self.realizations, self.curr_realizations)
        self.curr_realizations = self.new_relizations
        self.new_realizations = dict()

    def __iter__(self):
        for high_intervention in self.high_interventions:
            low_interventions = self.get_low_interventions()
            for low_intervention in low_interventions:
                yield {"low_intervention":low_intervention, "high_intervention": high_intervention, "high_output":,"low_output":}


def test_mapping(low_model, high_model, low_model_type, dataset, num_inputs,
                 batch_size, mapping, unwanted_low_nodes):
    if low_model_type not in {"bert", "lstm"}:
        raise NotImplementedError(
            f"Does not support low model type {low_model_type}")
    if unwanted_low_nodes is None:
        unwanted_low_nodes = set()
    relevant_mappings = {node: loc for node, loc in mapping.items() \
                         if node not in unwanted_low_nodes}
    high_node = list(relevant_mappings.keys())[0]
    low_node = list(relevant_mappings[high_node].keys())[0]
    low_loc = relevant_mappings[high_node][low_node]

    if len(relevant_mappings) > 1:
        raise NotImplementedError(
            "Currently does not support more than one intermediate nodes")

    print("--- Testing mapping", relevant_mappings)
    print("    Getting base outputs")
    device = torch.device("cuda")
    subset = Subset(dataset, list(range(num_inputs)))

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    input_tuple_len = len(subset[0])
    icd = InterchangeDataset(input_tuple_len, low_loc)

    low_batch_dim = 1 if low_model_type == "lstm" and not getattr(low_model, "batch_first", True) else 0
    for i, input_tuple in enumerate(dataloader):
        high_input_tensor = input_tuple[-2]
        # if low_model_type == "bert":
        #     high_input_tensor = input_tuple[-2]
        # elif low_model_type == "lstm":
        #     high_input_tensor = torch.cat((input_tuple[0][:,:9],
        #                                   input_tuple[0][:,10:]), dim=1)

        high_base_key = [serialize(x) for x in high_input_tensor]
        high_input = antra.GraphInput.batched(
            {"input": high_input_tensor.T}, high_base_key)
        # high model uses batch_dim = 0 because all intermediate outputs are
        # in batch-first order.
        high_output = high_model.compute(high_input)
        high_hidden = high_model.compute_node(high_node, high_input)

        low_key = [serialize(x) for x in input_tuple[0]]
        low_input_tuple_for_graph = [x.to(device) for x in input_tuple]
        # if low_model_type == "bert":
        #     low_input_tuple_for_graph = [x.to(device) for x in input_tuple]
        # elif low_model_type == "lstm":
        #     low_input_tuple_for_graph = [input_tuple[0].T.to(device),
        #                                  input_tuple[-1].to(device)]

        low_input = antra.GraphInput.batched(
            {"input": low_input_tuple_for_graph}, low_key, batch_dim=low_batch_dim)
        low_output = low_model.compute(low_input)
        low_hidden = low_model.compute_node(low_node, low_input)

        icd.add_example(low_input_tuple=input_tuple,
                        low_outputs=low_output.tolist(),
                        low_hidden_values=low_hidden,
                        high_inputs=high_input_tensor,
                        high_outputs=high_output.tolist(),
                        high_hidden_values=high_hidden)

    print("    Running causal_abstraction experiments")
    intervention_dataloader = DataLoader(icd, batch_size=batch_size)

    print("low_hidden_values", len(icd.low_hidden_values))
    res_dict = {"base_i": [], "interv_i": [],
                "high_base_res": [], "low_base_res": [],
                "high_interv_res": [], "low_interv_res": []}
    count = 0
    for i, input_tuple in enumerate(intervention_dataloader):
        if i % 100 == 0:
            print(f"    > {count} / {num_inputs ** 2}")
        high_input = input_tuple[icd.idx_high_inputs]
        high_interv_value = input_tuple[icd.idx_high_hidden]

        high_base_key = [serialize(x) for x in high_input]
        high_base = antra.GraphInput.batched(
            {"input": high_input.T}, high_base_key, cache_results=False
        )
        high_interv_key = [(serialize(x), serialize(interv)) for x, interv in \
                           zip(high_input, high_interv_value)]
        high_intervention = antra.Intervention.batched(
            high_base, high_interv_key,
            intervention={high_node: high_interv_value},
        )

        high_base_res, high_interv_res = \
            high_model.intervene(high_intervention, store_cache=False)

        low_input_tensor = input_tuple[0]
        low_interv_value = input_tuple[icd.idx_low_hidden]
        low_base_key = [serialize(x) for x in low_input_tensor]
        low_input_tuple_for_graph = [x.to(device) for x in input_tuple[:icd.idx_low_hidden]]
        # if low_model_type == "bert":
        #     low_input_tuple_for_graph = [x.to(device) for x in input_tuple[:5]]
        # elif low_model_type == "lstm":
        #     low_input_tuple_for_graph = [input_tuple[0].T.to(device),
        #                                  input_tuple[-1].to(device)]

        low_base = antra.GraphInput.batched(
            {"input": low_input_tuple_for_graph}, low_base_key,
            cache_results=False, batch_dim=low_batch_dim)

        low_interv_key = [(serialize(x), serialize(interv)) for x, interv in \
                          zip(low_input_tensor, low_interv_value)]

        low_intervention = antra.Intervention.batched(
            low_base, low_interv_key,
            intervention={low_node: low_interv_value.to(device)},
            location={low_node: low_loc}, batch_dim=low_batch_dim
        )

        low_base_res, low_interv_res = \
            low_model.intervene(low_intervention, store_cache=False)

        # assert torch.all(low_base_res.to(torch.device("cpu")) == input_tuple[icd.idx_low_outputs])
        # assert torch.all(high_base_res == input_tuple[icd.idx_high_outputs])

        res_dict["base_i"].extend(input_tuple[icd.idx_base_i].tolist())
        res_dict["interv_i"].extend(input_tuple[icd.idx_interv_i].tolist())
        res_dict["low_base_res"].extend(low_base_res.tolist())
        res_dict["high_base_res"].extend(high_base_res.tolist())
        res_dict["low_interv_res"].extend(low_interv_res.tolist())
        res_dict["high_interv_res"].extend(high_interv_res.tolist())
        count += len(input_tuple[icd.idx_base_i].tolist())

    icd.del_hidden_values() # delete hidden values to save space
    res_dict["interchange_dataset"] = icd
    res_dict["mapping"] = mapping

    return res_dict



def find_abstractions_batch(low_model, high_model, low_model_type,
                            dataset, num_inputs, batch_size,
                            fixed_assignments, unwanted_low_nodes=None):
    print("Creating possible mappings")
    mappings = create_possible_mappings(low_model, high_model, fixed_assignments,
                                        unwanted_low_nodes)
    print(f"Got {len(mappings)} different mappings")
    result = []
    with torch.no_grad():
        for mapping in mappings:
            result.append(test_mapping(low_model, high_model, low_model_type,
                                       dataset, num_inputs, batch_size, mapping, unwanted_low_nodes))

    return result
