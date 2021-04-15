import torch
import copy
from collections import defaultdict

from antra import *
from antra import GraphInput
from antra.location import serialize_location, deserialize_location, SerializedLocationType, reduce_dim
from antra.interchange.mapping import create_possible_mappings, AbstractionMapping
from antra.utils import SerializableType, idx_by_dim

from torch.utils.data import IterableDataset, DataLoader

from typing import *

HighNodeName = NodeName
LowNodeName = NodeName

Realization = Dict[(LowNodeName, SerializedLocationType), Any]
RealizationMapping = Dict[(HighNodeName, SerializableType), List[Realization]]

def merge_realization_mappings(current: RealizationMapping, other: RealizationMapping):
    for high_node_and_loc in other.keys():
        if high_node_and_loc not in current:
            current[high_node_and_loc] = other[high_node_and_loc]
        else:
            current[high_node_and_loc].extend(other[high_node_and_loc])


class CausalAbstraction:
    def __init__(
            self,
            low_model: ComputationGraph,
            high_model: ComputationGraph,
            low_inputs: Sequence[GraphInput],
            high_inputs: Sequence[GraphInput],
            high_interventions: Sequence[Intervention],
            fixed_node_mapping: AbstractionMapping,
            high_to_low_keys: Optional[Dict[SerializableType, SerializableType]]=None,
            unwanted_low_nodes: Optional[Sequence[str]]=None,
            low_nodes_to_indices: Optional[Dict[LowNodeName, List[LocationType]]]=None,
            batch_size: int=1,
            device: Union[str, torch.device]=None
    ):
        """
        :param low_model:
        :param high_model:
        :param low_inputs:
        :param high_inputs:
        :param high_interventions:
        :param fixed_node_mapping:
        :param high_to_low_keys: A mapping from high- to low-level model
            inputs. By default, assume `low_inputs` and `high_inputs` are the
            same length and the mapping between them is same as they are ordered
            in the list.
        :param unwanted_low_nodes:
        :param low_nodes_to_indices:
        :param batch_size:
        :param device:
        """
        self.low_model = low_model
        self.high_model = high_model
        self.low_inputs = low_inputs
        self.high_inputs = high_inputs
        self.high_interventions = high_interventions
        self.mappings = create_possible_mappings(
            low_model, high_model, fixed_node_mapping, unwanted_low_nodes,
            low_nodes_to_indices)

        self.low_keys_to_inputs = {gi.keys: gi for gi in low_inputs}
        self.high_keys_to_inputs = {gi.keys: gi for gi in high_inputs}
        if not high_to_low_keys:
            assert len(low_inputs) == len(high_inputs)
            high_to_low_keys = {hi.keys : lo.keys for hi, lo in zip(low_inputs, high_inputs)}
        self.high_to_low_keys = high_to_low_keys
        self.high_keys_to_low_inputs = {hi_key: self.low_keys_to_inputs[lo_key]
                                        for hi_key, lo_key in high_to_low_keys.items()}

        self.batch_size = batch_size

        self.device = device

    @torch.no_grad()
    def find_abstractions(self):
        results = []
        for mapping in self.mappings:
            results.append(self.test_mapping(mapping))
        return results

    def test_mapping(self, mapping: AbstractionMapping):
        self.curr_mapping = mapping
        icd = InterchangeDataset(mapping, self, collate_fn=self.interv_collate_fn)
        icd_dataloader = icd.get_dataloader(batch_size=self.batch_size, shuffle=False)

        batched = self.batch_size > 1

        while True:
            new_realizations: RealizationMapping = {}
            for batch in icd_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, (torch.Tensor, Intervention, GraphInput)) else v
                         for k, v in batch.items()}

                low_intervention = batch["low_intervention"]
                high_intervention = batch["high_intervention"]

                actual_batch_size = low_intervention.get_batch_size()

                high_base_res, high_interv_res = self.high_model.intervene_all_nodes(high_intervention)
                low_base_res, low_interv_res = self.low_model.intervene_all_nodes(low_intervention)

                realizations = self._create_new_realizations(
                    low_intervention, high_intervention, high_interv_res, low_interv_res, actual_batch_size)

                merge_realization_mappings(new_realizations, realizations)

            # TODO: start from here

    def interv_collate_fn(self, batch):
        return default_collate_fn(batch)

    def _create_new_realizations(self, high_intervention: Intervention, low_intervention: Intervention,
                                 high_interv_res: Dict[str, Any], low_interv_res: Dict[str, Any],
                                 actual_batch_size: int) -> RealizationMapping:
        realization_mapping: RealizationMapping = defaultdict(list)

        for high_node, high_values in high_interv_res.items():
            if not high_intervention.is_empty() and high_node not in high_intervention.affected_nodes:
                continue
            realizations = [{}] * actual_batch_size
            for low_node, low_loc in self.curr_mapping[high_node].items():
                ser_singleton_low_loc = serialize_location(reduce_dim(low_loc, dim=low_intervention.batch_dim))
                low_values = low_interv_res[low_node][low_loc]
                for i in range(actual_batch_size):
                    realizations[i][(low_node, ser_singleton_low_loc)] = idx_by_dim(low_values, i, low_intervention.batch_dim)

            for i in range(actual_batch_size):
                ser_high_value = idx_by_dim(high_values, i, high_intervention.batch_dim)
                realization_mapping[(high_node, ser_high_value)].append(realizations[i])

        return realization_mapping




class InterchangeDataset(IterableDataset):
    def __init__(self, mapping: AbstractionMapping, data: CausalAbstraction, device=None, collate_fn=None):
        super(InterchangeDataset, self).__init__()
        self.low_inputs = data.low_inputs
        self.high_inputs = data.high_inputs
        self.low_keys_to_inputs = data.low_keys_to_inputs
        self.high_keys_to_inputs = data.high_keys_to_inputs
        self.high_keys_to_low_inputs = data.high_keys_to_low_inputs

        self.high_interventions = data.high_interventions

        self.low_outputs = []
        self.high_outputs = []
        self.all_realizations = {} # Maps a high variable and value (V_H, v_H) to vectors of low-values
        self.curr_realizations = {} # Maps a high variable and value (V_H, v_H) to vectors of low-values, but just the last round
        self.new_realizations = None # Maps a high variable and value (V_H, v_H) to vectors of low-values, but just the new round
        self.mapping = mapping

        self.device = torch.device("cpu") if device is None else device

        self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn

        # self.populate_dataset()

    @property
    def num_examples(self):
        return len(self.low_outputs)

    def update_realizations(self, new_realizations):
        self.all_realizations.update(self.curr_realizations)
        self.curr_realizations = new_realizations
        self.new_realizations = {}

    # def populate_dataset(self):
    #     self.examples = []
    #     for high_interv in self.high_interventions:
    #         low_intervs = self.get_low_interventions(high_interv)
    #         self.examples.extend((high_interv, low_interv) for low_interv in low_intervs)

    def get_low_interventions(self, high_intervention: Intervention) -> List[Intervention]:
        high_interv: GraphInput = high_intervention.intervention
        low_base = self.high_keys_to_low_inputs[high_intervention.base.key]

        all_partial_intervs: List[Realization] = [{}]
        for high_var_name, high_var_value in high_interv.values.items():
            new_partial_intervs: List[Realization] = [{}]
            for pi in all_partial_intervs:
                for realization in self.all_realizations[(high_var_name, high_var_value)]:
                    # realization is Dict[(low node name, serialized location), value]
                    pi_copy = copy.copy(pi)
                    pi_copy.update(realization)
                    new_partial_intervs.append(pi_copy)
                for realization in self.curr_realizations[(high_var_name, high_var_value)]:
                    pi_copy = copy.copy(pi)
                    pi_copy.update(realization)
                    pi_copy["accepted"] = True
                    new_partial_intervs.append(pi_copy)

            all_partial_intervs = new_partial_intervs

        low_interventions = []
        for pi_dict in all_partial_intervs:
            if "accept" not in pi_dict:
                continue
            else:
                del pi_dict["accept"]
                new_low_interv = self.get_intervention_from_realizations(low_base, pi_dict) # unbatched
                low_interventions.append(new_low_interv)

        return low_interventions

    def get_intervention_from_realizations(self, low_base: GraphInput, partial_interv_dict: Realization) -> Intervention:
        # partial_interv_dict may contain multiple entries with same node but different locations,
        # e.g {(nodeA, loc1): val, (nodeA, loc2): val}, need to find and merge them
        val_dict, loc_dict = defaultdict(list)

        for (node_name, ser_low_loc), val in partial_interv_dict.items():
            val_dict[node_name].append(val)
            loc_dict[node_name].append(deserialize_location(ser_low_loc))

        for node_name in val_dict.keys():
            if len(val_dict[node_name]) == 1:
                val_dict[node_name] = val_dict[node_name][0]
            if len(loc_dict[node_name]) == 1:
                loc_dict[node_name] = loc_dict[node_name][0]

        return Intervention(low_base, intervention=val_dict, location=loc_dict)



    def __iter__(self):
        for high_intervention in self.high_interventions:
            low_interventions = self.get_low_interventions(high_intervention)
            for low_intervention in low_interventions:
                yield {
                    "low_intervention": low_intervention,
                    "high_intervention": high_intervention,
                    "high_output": None,
                    "low_output": None
                }

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)


def default_collate_fn(batch):
    pass

def bert_input_collate_fn(batch):
    pass