import copy
from collections import defaultdict
from pprint import pprint

import torch
import torch.utils.data as data

from typing import *

from antra import *
import antra.location as location
import antra.utils as utils
from antra.realization import Realization, SerializedRealization
from antra.interchange.mapping import create_possible_mappings, AbstractionMapping
from antra.interchange.batched import InterchangeDataset, merge_realization_mappings, pack_interventions


HighNodeName = NodeName
LowNodeName = NodeName

# SerializedRealization=Tuple[Tuple[Tuple[LowNodeName, location.SerializedLocationType], SerializedType], ...]
# Realization = Dict[Tuple[LowNodeName, location.SerializedLocationType], Any]
RealizationMapping = Dict[Tuple[HighNodeName, SerializedType], List[Realization]]
RealizationRecord = Dict[Tuple[HighNodeName, SerializedType], Set[SerializedRealization]]


class CounterfactualTraining:
    accepted_result_format = ["simple", "equality"]
    def __init__(
            self,
            low_model: ComputationGraph,
            high_model: ComputationGraph,
            low_inputs: Sequence[GraphInput],
            high_inputs: Sequence[GraphInput],
            high_interventions: Sequence[Intervention],
            fixed_node_mapping: AbstractionMapping,
            high_to_low_keys: Optional[Dict[SerializedType, SerializedType]]=None,
            ignored_low_nodes: Optional[Sequence[str]]=None,
            low_nodes_to_indices: Optional[Dict[LowNodeName, List[LocationType]]]=None,
            batch_dim: int = 0,
            batch_size: int = 1,
            cache_interv_results: bool = False,
            cache_base_results: bool = True,
            result_format: str = "equality",
            store_low_interventions: bool = False,
            trace_realization_origins: bool = False,
            device: Optional[torch.device] = None
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
        :param ignored_low_nodes:
        :param low_nodes_to_indices: A mapping from low node names to a list of
            all the possible indices (LOCs) in that low node to intervene on
        :param batch_dim:
        :param batch_size:
        :param cache_interv_results:
        :param cache_base_results:
        :param result_format: type of results to return
        :param store_low_interventions: store all the low-level interventions
            that are generated in `self.low_keys_to_interventions`
        :param trace_realization_origins: whether to trace the origin of low
            intervention values
        :param device:
        """
        if result_format not in CounterfactualTraining.accepted_result_format:
            raise ValueError(f"Incorrect result format '{result_format}'. "
                             f"Must be in {CounterfactualTraining.accepted_result_format}.")
        self.low_model = low_model
        self.high_model = high_model
        self.low_inputs = low_inputs
        self.high_inputs = high_inputs
        self.high_interventions = high_interventions

        if not all(not interv.batched for interv in high_interventions):
            raise ValueError("Does not support batched interventions for `high_interventions` in CausalAbstraction constructor")

        self.high_intervention_range = self._get_high_intervention_range(high_interventions)

        self.mappings = create_possible_mappings(
            low_model, high_model, fixed_node_mapping, ignored_low_nodes,
            low_nodes_to_indices
        )

        self.low_keys_to_inputs = {gi.keys: gi for gi in low_inputs}
        self.high_keys_to_inputs = {gi.keys: gi for gi in high_inputs}
        if not high_to_low_keys:
            assert len(low_inputs) == len(high_inputs)
            high_to_low_keys = {hi.keys : lo.keys for lo, hi in zip(low_inputs, high_inputs)}
        self.high_to_low_input_keys = high_to_low_keys
        self.high_keys_to_low_inputs = {hi_key: self.low_keys_to_inputs[lo_key]
                                        for hi_key, lo_key in high_to_low_keys.items()}

        self.ignored_low_nodes = {self.low_model.root} | self.low_model.leaves
        if ignored_low_nodes is not None:
            self.ignored_low_nodes |= set(ignored_low_nodes)

        self.result_format = result_format
        self.store_low_interventions = store_low_interventions
        self.trace_realization_origins = trace_realization_origins

        self.high_keys_to_interventions = {ivn.keys: ivn for ivn in high_interventions}
        self.low_keys_to_interventions = {}

        self._add_empty_interventions()

        # following info used to set up batched interventions
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.cache_interv_results = cache_interv_results
        self.cache_base_results = cache_base_results

        self.high_key_leaves = high_inputs[0].key_leaves
        self.low_key_leaves = low_inputs[0].key_leaves
        self.high_non_batch_leaves = high_inputs[0].non_batch_leaves
        self.low_non_batch_leaves = low_inputs[0].non_batch_leaves

        self.device = device if device is not None else torch.device("cpu")

    def _get_high_intervention_range(self, high_interventions: Sequence[Intervention])\
            -> Dict[HighNodeName, Set[SerializedType]]:
        interv_range = defaultdict(set)
        for interv in high_interventions:
            for high_node_name, val in interv.intervention._values.items():
                interv_range[high_node_name].add(utils.serialize(val))
        return interv_range

    def _add_empty_interventions(self):
        all_empty_ivns = []
        for high_input in self.high_inputs:
            empty_ivn = Intervention(high_input, intervention={})
            if empty_ivn.keys not in self.high_keys_to_interventions:
                all_empty_ivns.append(empty_ivn)
                self.high_keys_to_interventions[empty_ivn.keys] = empty_ivn
        self.high_interventions = all_empty_ivns + self.high_interventions

    def has_high_intervention(self, high_node_name: str, val: Any):
        ser_val = utils.serialize(val) if not utils.is_serialized(val) else val
        return ser_val in self.high_intervention_range[high_node_name]

    def find_abstractions(self):
        results = []
        for mapping in self.mappings:
            results.append((self.train(mapping), mapping))
        return results

    def train(self, mapping: AbstractionMapping) -> Dict[Tuple[SerializedType, SerializedType], Any]:
        self.curr_mapping: AbstractionMapping = mapping

        icd = InterchangeDataset(mapping, self, collate_fn=self.collate_fn)
        random_batch_sampler = data.RandomSampler(data.BatchSampler(data.SequentialSampler(icd), batch_size=self.batch_size, drop_last=False))
        icd_dataloader = icd.get_dataloader(sampler=random_batch_sampler)

        optimizer = torch.optim.Adam(self.low_model.model.params())
        loss_fxn = torch.nn.NLLLoss()

        results = {}
        iteration = 0
        while True:
            # print(f"======== iteration {iteration} ========")
            new_realizations: RealizationMapping = {}
            total_new_realizations = 0
            num_interventions = 0
            for batch in icd_dataloader:
                # batch is a list of dicts, each dict contains a batched low and high intervention

                pprint(batch)
                for minibatch in batch:
                    optimizer.zero_grad()

                    minibatch = {k: v.to(self.device) \
                        if isinstance(v, (torch.Tensor, Intervention, GraphInput)) else v
                        for k, v in minibatch.items()}

                    low_intervention = minibatch["low_intervention"]
                    high_intervention = minibatch["high_intervention"]
                    # print("low interv")
                    # print(low_intervention.base)
                    # print("high interv")
                    # pprint(high_intervention.base)

                    actual_batch_size = low_intervention.get_batch_size()
                    num_interventions += actual_batch_size

                    high_base_res, high_ivn_res = self.high_model.intervene_all_nodes(high_intervention)
                    low_base_res, low_ivn_res = self.low_model.intervene_all_nodes(low_intervention)

                    logits = low_ivn_res[self.low_model.root.name]
                    labels = high_ivn_res[self.high_model.root.name]

                    loss = loss_fxn(logits, labels)
                    loss.backward()
                    optimizer.step()

                    if not high_intervention.is_empty():
                        self._add_results(
                            results, high_intervention, low_intervention,
                                high_base_res, high_ivn_res, low_base_res, low_ivn_res,
                                actual_batch_size
                        )

                    realizations, num_new_realizations = \
                        self._create_new_realizations(
                            icd, high_intervention, low_intervention,
                            high_ivn_res, low_ivn_res, actual_batch_size)

                    total_new_realizations += num_new_realizations
                    merge_realization_mappings(new_realizations, realizations)

            print(f"Got {total_new_realizations} new realizations")
            # print(new_realizations)
            if iteration == 0:
                icd.did_empty_interventions = True
            iteration += 1
            if total_new_realizations == 0:
                break
            else:
                icd.update_realizations(new_realizations)

        return results

    def _add_results(self, results: Dict, high_ivn: Intervention, low_ivn: Intervention,
                     high_base_res: Dict[str, Any], high_ivn_res: Dict[str, Any],
                     low_base_res: Dict[str, Any], low_ivn_res: Dict[str, Any],
                     actual_batch_size: int):
        low_root = self.low_model.root.name
        high_root = self.high_model.root.name
        hi_base_res, hi_ivn_res = high_base_res[high_root], high_ivn_res[high_root]
        lo_base_res, lo_ivn_res = low_base_res[low_root], low_ivn_res[low_root]

        for i in range(actual_batch_size):
            key = (low_ivn.keys[i], high_ivn.keys[i])
            if self.result_format == "simple":
                _hi_res = utils.idx_by_dim(hi_ivn_res, i, high_ivn.batch_dim)
                _lo_res = utils.idx_by_dim(lo_ivn_res, i, low_ivn.batch_dim)
                results[key] = (_hi_res == _lo_res)
            elif self.result_format == "equality":
                _hi_base_res = utils.idx_by_dim(hi_base_res, i, high_ivn.batch_dim)
                _hi_ivn_res = utils.idx_by_dim(hi_ivn_res, i, high_ivn.batch_dim)
                _lo_base_res = utils.idx_by_dim(lo_base_res, i, low_ivn.batch_dim)
                _lo_ivn_res = utils.idx_by_dim(lo_ivn_res, i, low_ivn.batch_dim)
                d = {
                    "base_eq": _hi_base_res == _lo_base_res,
                    "ivn_eq": _hi_ivn_res == _lo_ivn_res,
                    "low_base_eq_ivn": _lo_base_res == _lo_ivn_res,
                    "high_base_eq_ivn": _hi_base_res == _hi_ivn_res
                }
                results[key] = d
            else:
                raise ValueError(f"Invalid result_format: {self.result_format}")

    def _create_new_realizations(self, icd: "InterchangeDataset",
                                 high_ivn_batch: Intervention, low_ivn_batch: Intervention,
                                 high_ivn_res: Dict[str, Any], low_ivn_res: Dict[str, Any],
                                 actual_batch_size: int) -> (RealizationMapping, int):
        rzn_mapping: RealizationMapping = defaultdict(list)
        record: RealizationRecord = icd.realization_record
        new_rzn_count = 0
        # print("--- Creating new rzns")

        # TODO: Keep track of where the interventions came from
        origins = [self.low_keys_to_interventions[low_ivn_batch.keys[i]] \
                   for i in range(actual_batch_size)] if self.trace_realization_origins else None

        for high_node, high_values in high_ivn_res.items():
            if high_node not in self.high_intervention_range:
                continue
            if not high_ivn_batch.is_empty() and high_node not in high_ivn_batch.affected_nodes:
                continue

            rzns = [Realization() for _ in range(actual_batch_size)]
            # print("\nnum low mappings", len(self.curr_mapping[high_node]))
            for low_node, low_loc_list in self.curr_mapping[high_node].items():
                if not isinstance(low_loc_list, list):
                    low_loc_list = [low_loc_list]
                for low_loc in low_loc_list:
                    ser_low_loc = location.serialize_location(low_loc)
                    low_values = low_ivn_res[low_node] if low_loc is None else low_ivn_res[low_node][low_loc]
                    key = (low_node, ser_low_loc)
                    # print(f"low_values, shape={low_values.shape}")
                    # print(low_values)
                    for i in range(actual_batch_size):
                        _low_val = utils.idx_by_dim(low_values, i, low_ivn_batch.batch_dim)
                        if origins:
                            rzns[i].add(key, _low_val, origins[i])
                        else:
                            rzns[i][key] = _low_val

            for i in range(actual_batch_size):
                high_value = utils.idx_by_dim(high_values, i, high_ivn_batch.batch_dim)
                ser_high_value = utils.serialize(high_value)

                if not ser_high_value in self.high_intervention_range[high_node]:
                    continue

                # before finally adding realization to the new realizations,
                # check if we've seen it before
                ser_rzn = rzns[i].serialize()
                if ser_rzn in record[(high_node, ser_high_value)]:
                    continue

                record[(high_node, ser_high_value)].add(ser_rzn)
                rzn_mapping[(high_node, ser_high_value)].append(rzns[i])
                new_rzn_count += 1

        return rzn_mapping, new_rzn_count

    def collate_fn(self, batch: List[Dict]) -> List[Dict]:
        """ Collate function called by dataloader.

        :param batch:
        :return:
        """
        # package up a list of individual interventions into multiple batched interventions
        # batch may contain interventions on different locations
        high_node_to_minibatches = defaultdict(list)
        for d in batch:
            high_nodes = tuple(sorted(d["high_intervention"].intervention._values.keys()))
            high_node_to_minibatches[high_nodes].append(d)

        minibatches = []
        for minibatch_dicts in high_node_to_minibatches.values():
            low_base_dict, low_ivn_dict, low_loc_dict = pack_interventions(
                [d["low_intervention"] for d in minibatch_dicts],
                batch_dim=self.batch_dim,
                non_batch_inputs=self.low_non_batch_leaves
            )
            low_base_input = GraphInput(
                low_base_dict, batched=True, batch_dim=self.batch_dim,
                cache_results=self.cache_base_results,
                key_leaves=self.low_key_leaves,
                non_batch_leaves=self.low_non_batch_leaves
            )
            low_ivn = Intervention.batched(
                low_base_input, low_ivn_dict, low_loc_dict,
                batch_dim=self.batch_dim, cache_base_results=self.cache_interv_results)

            high_base_dict, high_ivn_dict, high_loc_dict = pack_interventions(
                [d["high_intervention"] for d in minibatch_dicts],
                batch_dim=self.batch_dim,
                non_batch_inputs=self.high_non_batch_leaves
            )
            high_base_input = GraphInput(
                high_base_dict, batched=True, batch_dim=self.batch_dim,
                cache_results=self.cache_base_results,
                key_leaves=self.high_key_leaves,
                non_batch_leaves=self.high_non_batch_leaves
            )
            high_ivn = Intervention.batched(
                high_base_input, high_ivn_dict, high_loc_dict,
                batch_dim=self.batch_dim, cache_base_results=self.cache_interv_results)

            minibatches.append({"low_intervention": low_ivn,
                                "high_intervention": high_ivn})

        return minibatches


