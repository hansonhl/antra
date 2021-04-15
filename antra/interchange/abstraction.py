import torch
import copy
from collections import defaultdict

from antra import *
import antra.location as location
import antra.utils as utils

from antra.interchange.mapping import create_possible_mappings, AbstractionMapping
# from antra.utils import serialize, is_serialized, idx_by_dim, SerializedType

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from typing import *

HighNodeName = NodeName
LowNodeName = NodeName

Realization = Dict[(LowNodeName, location.SerializedLocationType), Any]
RealizationMapping = Dict[(HighNodeName, SerializedType), List[Realization]]


class CausalAbstraction:
    def __init__(
            self,
            low_model: ComputationGraph,
            high_model: ComputationGraph,
            low_inputs: Sequence[GraphInput],
            high_inputs: Sequence[GraphInput],
            high_interventions: Sequence[Intervention],
            fixed_node_mapping: AbstractionMapping,
            high_to_low_keys: Optional[Dict[SerializedType, SerializedType]]=None,
            unwanted_low_nodes: Optional[Sequence[str]]=None,
            low_nodes_to_indices: Optional[Dict[LowNodeName, List[LocationType]]]=None,
            batch_dim: int = 0,
            batch_size: int = 1,
            cache_interv_results: bool = False,
            cache_base_results: bool = True,
            non_batch_leaves: Sequence[str] = None,
            device: Union[str, torch.device] = None
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
        :param low_nodes_to_indices: A mapping from low node names to a list of
            all the possible indices (LOCs) in that low node to intervene on
        :param batch_dim:
        :param batch_size:
        :param cache_interv_results:
        :param cache_base_results:
        :param non_batch_leaves:
        :param device:
        """
        self.low_model = low_model
        self.high_model = high_model
        self.low_inputs = low_inputs
        self.high_inputs = high_inputs
        self.high_interventions = high_interventions

        if not all(not interv.batched for interv in high_interventions):
            raise ValueError("Does not support batched interventions for `high_interventions` in CausalAbstraction constructor")

        self.high_intervention_range = self._get_high_intervention_range(high_interventions)

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

        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.cache_interv_results = cache_interv_results
        self.cache_base_results = cache_base_results
        self.non_batch_leaves = non_batch_leaves

        self.device = device

    def _get_high_intervention_range(self, high_interventions: Sequence[Intervention])\
            -> Dict[HighNodeName, Set[SerializedType]]:
        interv_range = defaultdict(set)
        for interv in high_interventions:
            for high_node_name, val in interv.intervention.values:
                interv_range[high_node_name].add(utils.serialize(val))
        return interv_range

    def has_high_intervention(self, high_node_name: str, val: Any):
        ser_val = utils.serialize(val) if not utils.is_serialized(val) else val
        return ser_val in self.high_intervention_range[high_node_name]

    @torch.no_grad()
    def find_abstractions(self):
        results = []
        for mapping in self.mappings:
            results.append(self.test_mapping(mapping))
        return results

    def test_mapping(self, mapping: AbstractionMapping):
        self.curr_mapping = mapping
        icd = InterchangeDataset(mapping, self, collate_fn=self.collate_fn)
        icd_dataloader = icd.get_dataloader(batch_size=self.batch_size, shuffle=False)

        batched = self.batch_size > 1

        while True:
            new_realizations: RealizationMapping = {}
            total_new_realizations = 0
            for batch in icd_dataloader:
                # batch is a list of dicts, each dict contains a batched low and high intervention
                for minibatch in batch:
                    minibatch = {k: v.to(self.device) \
                        if isinstance(v, (torch.Tensor, Intervention, GraphInput)) else v
                        for k, v in minibatch.items()}

                    low_intervention = minibatch["low_intervention"]
                    high_intervention = minibatch["high_intervention"]

                    actual_batch_size = low_intervention.get_batch_size()

                    high_base_res, high_interv_res = self.high_model.intervene_all_nodes(high_intervention)
                    low_base_res, low_interv_res = self.low_model.intervene_all_nodes(low_intervention)

                    realizations, num_new_realizations = self._create_new_realizations(
                        low_intervention, high_intervention, high_interv_res, low_interv_res, actual_batch_size)

                    total_new_realizations += num_new_realizations
                    merge_realization_mappings(new_realizations, realizations)
            if total_new_realizations == 0:
                break
            else:
                icd.update_realizations(new_realizations)


    def _create_new_realizations(self, high_intervention: Intervention, low_intervention: Intervention,
                                 high_interv_res: Dict[str, Any], low_interv_res: Dict[str, Any],
                                 actual_batch_size: int) -> (RealizationMapping, int):
        realization_mapping: RealizationMapping = defaultdict(list)
        num_new_realizations = 0

        for high_node, high_values in high_interv_res.items():
            if high_intervention.is_empty() \
                    or high_node in high_intervention.affected_nodes \
                    and high_node in self.high_intervention_range:
                realizations = [{}] * actual_batch_size
                for low_node, low_loc in self.curr_mapping[high_node].items():
                    ser_singleton_low_loc = location.serialize_location(
                        location.reduce_dim(low_loc, dim=low_intervention.batch_dim))
                    low_values = low_interv_res[low_node][low_loc]
                    for i in range(actual_batch_size):
                        realizations[i][(low_node, ser_singleton_low_loc)] = \
                            utils.idx_by_dim(low_values, i, low_intervention.batch_dim)

                for i in range(actual_batch_size):
                    high_value = utils.idx_by_dim(high_values, i, high_intervention.batch_dim)
                    ser_high_value = utils.serialize(high_value)
                    if ser_high_value in self.high_intervention_range[high_node]:
                        realization_mapping[(high_node, ser_high_value)].append(realizations[i])
                        num_new_realizations += 1

        return realization_mapping, num_new_realizations

    def collate_fn(self, batch: List[Dict]) -> List[Dict]:
        """ Collate function called by dataloader.

        :param batch:
        :return:
        """
        # package up a list of individual interventions into multiple batched interventions
        # batch may contain interventions on different locations
        high_node_to_minibatches = defaultdict(list)
        for d in batch:
            high_nodes = tuple(sorted(d["high_intervention"].intervention.values.keys()))
            high_node_to_minibatches[high_nodes].append(d)
        minibatches = []
        for minibatch_dicts in high_node_to_minibatches.values():
            low_interv = pack_interventions(
                [d["low_intervention"] for d in minibatch_dicts],
                batch_dim=self.batch_dim,
                cache_results=self.cache_interv_results,
                cache_base_results=self.cache_base_results,
                non_batch_inputs=self.non_batch_leaves
            )

            high_interv = pack_interventions(
                [d["high_intervention"] for d in minibatch_dicts],
                batch_dim=self.batch_dim,
                cache_results=self.cache_interv_results,
                cache_base_results=self.cache_base_results,
                non_batch_inputs=self.non_batch_leaves
            )
            minibatches.append({"low_intervention": low_interv,
                                "high_intervention": high_interv})

        return minibatches



class InterchangeDataset(IterableDataset):
    def __init__(self, mapping: AbstractionMapping, data: CausalAbstraction,
                 collate_fn: Callable):
        super(InterchangeDataset, self).__init__()
        self.low_inputs = data.low_inputs
        self.high_inputs = data.high_inputs
        self.low_keys_to_inputs = data.low_keys_to_inputs
        self.high_keys_to_inputs = data.high_keys_to_inputs
        self.high_keys_to_low_inputs = data.high_keys_to_low_inputs

        self.high_interventions = data.high_interventions

        self.low_outputs = []
        self.high_outputs = []
        self.all_realizations: RealizationMapping = defaultdict(list) # Maps a high variable and value (V_H, v_H) to vectors of low-values
        self.curr_realizations: RealizationMapping = defaultdict(list) # Maps a high variable and value (V_H, v_H) to vectors of low-values, but just the last round

        self.mapping = mapping
        self.collate_fn = collate_fn
        # self.populate_dataset()

    @property
    def num_examples(self):
        return len(self.low_outputs)

    def update_realizations(self, new_realizations):
        merge_realization_mappings(self.all_realizations, new_realizations)
        self.curr_realizations = new_realizations

    # def populate_dataset(self):
    #     self.examples = []
    #     for high_interv in self.high_interventions:
    #         low_intervs = self.get_low_interventions(high_interv)
    #         self.examples.extend((high_interv, low_interv) for low_interv in low_intervs)

    def _get_realizations(self, high_intervention: Intervention) -> List[Realization]:
        high_interv: GraphInput = high_intervention.intervention
        # low_base = self.high_keys_to_low_inputs[high_intervention.base.key]

        all_realizations: List[Realization] = [{}]

        for high_var_name, high_var_value in high_interv.values.items():
            new_partial_intervs: List[Realization] = [{}]
            for pi in all_realizations:
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

            all_realizations = new_partial_intervs

        return all_realizations

    # def get_low_interventions(self, high_intervention: Intervention) -> List[Intervention]:
    #     high_interv: GraphInput = high_intervention.intervention
    #     low_base = self.high_keys_to_low_inputs[high_intervention.base.key]
    #
    #     all_partial_intervs: List[Realization] = [{}]
    #     for high_var_name, high_var_value in high_interv.values.items():
    #         new_partial_intervs: List[Realization] = [{}]
    #         for pi in all_partial_intervs:
    #             for realization in self.all_realizations[(high_var_name, high_var_value)]:
    #                 # realization is Dict[(low node name, serialized location), value]
    #                 pi_copy = copy.copy(pi)
    #                 pi_copy.update(realization)
    #                 new_partial_intervs.append(pi_copy)
    #             for realization in self.curr_realizations[(high_var_name, high_var_value)]:
    #                 pi_copy = copy.copy(pi)
    #                 pi_copy.update(realization)
    #                 pi_copy["accepted"] = True
    #                 new_partial_intervs.append(pi_copy)
    #
    #         all_partial_intervs = new_partial_intervs
    #
    #     low_interventions = []
    #     for pi_dict in all_partial_intervs:
    #         if "accept" not in pi_dict:
    #             continue
    #         else:
    #             del pi_dict["accept"]
    #             new_low_interv = self.get_intervention_from_realizations(low_base, pi_dict) # unbatched
    #             low_interventions.append(new_low_interv)
    #
    #     return low_interventions

    def get_intervention_from_realizations(self, low_base: GraphInput, partial_interv_dict: Realization) -> Intervention:
        # partial_interv_dict may contain multiple entries with same node but different locations,
        # e.g {(nodeA, loc1): val, (nodeA, loc2): val}, need to find and merge them
        val_dict, loc_dict = defaultdict(list)

        for (node_name, ser_low_loc), val in partial_interv_dict.items():
            val_dict[node_name].append(val)
            loc_dict[node_name].append(location.deserialize_location(ser_low_loc))

        for node_name in val_dict.keys():
            if len(val_dict[node_name]) == 1:
                val_dict[node_name] = val_dict[node_name][0]
            if len(loc_dict[node_name]) == 1:
                loc_dict[node_name] = loc_dict[node_name][0]

        return Intervention(low_base, intervention=val_dict, location=loc_dict)


    # TODO: Yield while getting the realizations
    def __iter__(self):
        for high_intervention in self.high_interventions:
            # low_interventions = self.get_low_interventions(high_intervention)
            # for low_intervention in low_interventions:
            low_base = self.high_keys_to_low_inputs[high_intervention.base.key]
            realizations = self._get_realizations(high_intervention)
            for r_dict in realizations:
                if "accept" not in r_dict:
                    continue
                else:
                    del r_dict["accept"]
                    low_intervention = self.get_intervention_from_realizations(low_base, r_dict)

                    yield {
                        "low_intervention": low_intervention,
                        "high_intervention": high_intervention,
                        # "high_output": None,
                        # "low_output": None
                    }

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)


def merge_realization_mappings(current: RealizationMapping, other: RealizationMapping):
    for high_node_and_val in other.keys():
        if high_node_and_val not in current:
            current[high_node_and_val] = other[high_node_and_val]
        else:
            current[high_node_and_val].extend(other[high_node_and_val])



def pack_interventions(
        interventions: Sequence[Intervention],
        batch_dim: int = 0,
        cache_results: bool = False,
        cache_base_results: bool = True,
        non_batch_inputs: Optional[set] = None
    ) -> Intervention:
    """ Pack a sequence of individual interventions into one batched intervention.
    All interventions must be in the same nodes and locations.

    :param interventions:
    :param batch_dim:
    :param cache_results:
    :param cache_base_results:
    :param non_batch_inputs:
    :return:
    """
    base_lists, interv_lists = defaultdict(list), defaultdict(list)
    loc_dict = {}
    multi_loc_nodes = set()
    batch_size = len(interventions)
    interv_is_empty = False

    for interv in interventions:
        for leaf, val in interv.base.values:
            base_lists[leaf].append(val)

        if interv_is_empty and not interv.is_empty():
            raise RuntimeError(f"Cannot pack empty interventions together with non-empty ones")
        if interv.is_empty(): interv_is_empty = True

        for node, val in interv.intervention.values:
            if not isinstance(val, list):
                assert node not in multi_loc_nodes
                interv_lists[node].append(val)
            else:
                # multi-loc interventions
                if node not in interv:
                    multi_loc_nodes.add(node)
                    interv_lists[node] = [[] for _ in range(len(val))]
                else:
                    assert node in multi_loc_nodes
                assert len(val) == len(interv_lists[node])
                for i in range(len(val)):
                    interv_lists[node][i].append(val[i])

        for node, loc in interv.location:
            if node not in loc_dict:
                loc_dict[node] = location.expand_dim(loc, batch_dim)
            else:
                if location.expand_dim(loc) != loc_dict[node]:
                    raise RuntimeError(f"Locs do not match in the sequence of interventions! "
                                       f"(found both {loc} and {loc_dict[node]} for node {node})")

    # make sure base lists have equal length
    if not all(len(l) == batch_size for l in base_lists.values()):
        for leaf, vals in base_lists.items():
            if len(vals) != batch_size:
                raise RuntimeError(f"List of values for leaf `{leaf}` has shorter length ({len(vals)}) than batch size ({batch_size})")

    # make sure intervention values have equal length
    if not interv_is_empty:
        for node, vals in interv_lists.items():
            if node not in multi_loc_nodes:
                if len(vals) != batch_size:
                    raise RuntimeError(f"List of values for intervention at `{node}` has shorter length ({len(vals)}) than batch size ({batch_size})")
            else:
                if not all(len(vals[j]) == batch_size for j in range(len(vals))):
                    raise RuntimeError(f"Lists of values for multi-location intervention have inconsistent length")


    base_dict = batchify(base_lists, batch_dim, non_batch_inputs)
    interv_dict = batchify(interv_lists, batch_dim) if not interv_is_empty else {}

    return Intervention.batched(base_dict, interv_dict, loc_dict,
                                cache_results=cache_results,
                                cache_base_results=cache_base_results)

def batchify(input_lists, batch_dim, non_batch_inputs=None):
    input_dict = {}
    for key, vals in input_lists.items():
        one_val = vals[0]
        if non_batch_inputs and key in non_batch_inputs:
            input_dict[key] = one_val
        elif utils.is_torch_tensor(one_val):
            input_dict[key] = torch.stack(vals, dim=batch_dim)
        elif utils.is_numpy_array(one_val):
            input_dict[key] = np.stack(vals, axis=batch_dim)
        else:
            raise RuntimeError(f"Currently does not support automatically batchifying inputs with type {type(one_val)} for node `{key}`")
    return input_dict


def bert_input_collate_fn(batch):
    pass