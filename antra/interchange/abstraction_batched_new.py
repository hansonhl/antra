# This script is the same as `causal_abstraction/interchange.py`.
# It is duplicated here to support old pickled experiment result files.

import torch
import copy

from antra import *
from antra import GraphInput
from antra.location import serialize_location, deserialize_location
from antra.interchange.mapping import create_possible_mappings, AbstractionMappingType
from antra.utils import SerializableType

from torch.utils.data import IterableDataset, DataLoader

from typing import *


class CausalAbstraction:
    def __init__(
            self,
            low_model: ComputationGraph,
            high_model: ComputationGraph,
            low_inputs: Sequence[GraphInput],
            high_inputs: Sequence[GraphInput],
            high_interventions: Sequence[Intervention],
            fixed_node_mapping: AbstractionMappingType,
            high_to_low_keys: Optional[Dict[SerializableType, SerializableType]]=None,
            unwanted_low_nodes: Optional[Sequence[str]]=None,
            low_nodes_to_indices: Optional[Dict[NodeName, List[LocationType]]]=None,
            interv_collate_fn: Union[str, Callable, None]=None,
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
        :param interv_collate_fn:
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

        if isinstance(interv_collate_fn, str):
            if interv_collate_fn == "bert":
                self.interv_collate_fn = bert_input_collate_fn
            else:
                raise ValueError(f"Invalid name for interv_collate_fn: {interv_collate_fn}")
        else:
            self.interv_collate_fn = interv_collate_fn

        self.device = device

    @torch.no_grad()
    def find_abstractions(self):
        results = []
        for mapping in self.mappings:
            results.append(self.test_mapping(mapping))
        return results

    def test_mapping(self, mapping: AbstractionMappingType):
        icd = InterchangeDataset(mapping, self, collate_fn=self.interv_collate_fn)
        icd_dataloader = icd.get_dataloader(batch_size=self.batch_size, shuffle=False)

        batched = self.batch_size > 1

        while True:
            new_realizations = {}
            for batch in icd_dataloader:
                batch = self.prepare_batch(batch)

                low_intervention = batch["low_intervention"]
                high_intervention = batch["high_intervention"]

                high_base_res, high_interv_res = self.high_model.intervene_all_nodes(high_intervention)
                low_base_res, low_interv_res = self.low_model.intervene_all_nodes(low_intervention)

                return high_base_res, high_interv_res, low_base_res, low_interv_res

                realizations, realization_inputs = self._create_new_realizations()

    def prepare_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def _create_new_realizations(self, high_intervention, low_intervention, high_interv_res, low_interv_res, actual_batch_size):
        realizations: Dict[Tuple[NodeName, Any], Any] = {}
        realization_inputs: Dict[Tuple[NodeName, Any], Intervention] = {}

        if actual_batch_size > 1:
            pass
        else:
            raise NotImplementedError

        return realizations, realization_inputs




class InterchangeDataset(IterableDataset):
    def __init__(self, mapping: AbstractionMappingType, data: CausalAbstraction, device=None, collate_fn=None):
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

        all_partial_intervs = [{}]
        for high_var_name, high_var_value in high_interv.values.items():
            low_var_name, low_loc = list(self.mapping[high_var_name].items())[0]
            serialized_low_loc = serialize_location(low_loc)

            new_partial_intervs = [{}]
            for pi in all_partial_intervs:
                for realization in self.all_realizations[(high_var_name, high_var_value)]:
                    pi_copy = copy.copy(pi)

                    pi_copy[(low_var_name, serialized_low_loc)] = realization
                    new_partial_intervs.append(pi_copy)
                for realization in self.curr_realizations[(high_var_name, high_var_value)]:
                    pi_copy = copy.copy(pi)
                    pi_copy[(low_var_name, serialized_low_loc)] = realization
                    pi_copy["accepted"] = True
                    new_partial_intervs.append(pi_copy)

            all_partial_intervs = new_partial_intervs

        low_interventions = []
        for interv_dict in all_partial_intervs:
            if "accept" not in interv_dict:
                continue
            else:
                del interv_dict["accept"]
                interv_dict = {
                    low_var_name: val
                    for (low_var_name, _), val in interv_dict.items()
                }
                loc_dict = {
                    low_var_name: deserialize_location(low_serialized_loc)
                    for low_var_name, low_serialized_loc in interv_dict.keys()
                }
                new_low_interv = Intervention(low_base, intervention=interv_dict, location=loc_dict)
                low_interventions.append(new_low_interv)

        return low_interventions

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