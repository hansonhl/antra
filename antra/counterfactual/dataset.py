import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

import antra
from antra.interchange.batched import pack_interventions, pack_graph_inputs
from antra.interchange.mapping import AbstractionMapping

from typing import *


class ListCounterfactualDataset(Dataset):
    """Dataset of counterfactual training at one location """
    def __init__(
            self,
            base_dataset: Dataset,
            intervention_pairs: List[Tuple[int, int]],
            mapping: AbstractionMapping,
            intervention_weight: Optional[Union[Callable, List[float]]] = None,
            batch_dim: int = 0
    ):
        super(ListCounterfactualDataset, self).__init__()

        # noinspection PyTypeChecker
        if not isinstance(base_dataset, Dataset):
            raise ValueError("`base_dataset` must be of type `torch.utils.data.Dataset`")

        self.base_dataset_len = len(base_dataset)
        self.base_dataset = base_dataset
        self.intervention_pairs = intervention_pairs
        self.intervention_weight = intervention_weight
        if isinstance(intervention_weight, list) and len(intervention_weight) != len(self.intervention_pairs):
            raise ValueError(f"If given as a list, `intervention_weight` must "
                             f"have the same length as `intervention_pairs`. "
                             f"Given {len(intervention_weight)}, "
                             f"expected{len(intervention_pairs)}")
        self.mapping = mapping
        self.batch_dim = batch_dim

    def __len__(self):
        return len(self.intervention_pairs)

    def __getitem__(self, item):
        base_idx, ivn_src_idx = self.intervention_pairs[item]
        base = self.base_dataset[base_idx]
        ivn_src = self.base_dataset[ivn_src_idx]
        high_ivn = self.construct_intervention(base, ivn_src)
        weight = None
        if self.intervention_weight is not None:
            if isinstance(self.intervention_weight, list):
                weight = self.intervention_weight[item]
            else:
                weight = self.intervention_weight(high_ivn)

        low_base_input = self.construct_low_input(base)
        low_ivn_src = self.construct_low_input(ivn_src)

        return {
            "high_intervention": high_ivn,
            "low_base_input": low_base_input,
            "low_intervention_source": low_ivn_src,
            "weight": weight
        }

    def construct_low_input(self, example) -> antra.GraphInput:
        """ Construct a GraphInput object for the low level model"""
        raise NotImplementedError

    def construct_intervention(self, base, ivn_source) -> antra.Intervention:
        """ Construct the high level intervention object given two examples.

        :param base:
        :param ivn_source:
        """
        raise NotImplementedError

    def collate_fn(self, batch):
        return cf_collate_fn(batch, self.mapping, batch_dim=self.batch_dim)

class RandomCounterfactualDataset(IterableDataset):
    def __init__(
            self,
            base_dataset: Dataset,
            mapping: AbstractionMapping,
            intervention_weight_fn: Optional[Callable] = None,
            batch_dim: int = 0,
            num_random_bases=50000,
            num_random_ivn_srcs=20,
            fix_examples=False
    ):
        super(RandomCounterfactualDataset, self).__init__()

        # noinspection PyTypeChecker
        if not isinstance(base_dataset, Dataset):
            raise ValueError("`base_dataset` must be of type `torch.utils.data.Dataset`")

        self.base_dataset_len = len(base_dataset)
        self.base_dataset = base_dataset
        self.intervention_weight_fn = intervention_weight_fn
        self.num_random_bases = num_random_bases
        self.num_random_ivn_srcs = num_random_ivn_srcs
        self.mapping = mapping
        self.batch_dim = batch_dim

        self.num_examples = num_random_bases * num_random_ivn_srcs
        self.fix_examples = fix_examples

        self.base_idxs_to_ivn_src_idxs = {}
        if self.fix_examples:
            self.rand_base_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_bases]


    def __iter__(self):
        if not self.fix_examples:
            rand_base_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_bases]
        else:
            rand_base_idxs = self.rand_base_idxs

        for base_idx in rand_base_idxs:
            base_idx = base_idx.item()
            if not self.fix_examples:
                ivn_src_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_ivn_srcs]
            else:
                if base_idx not in self.base_idxs_to_ivn_src_idxs:
                    ivn_src_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_ivn_srcs]
                    self.base_idxs_to_ivn_src_idxs[base_idx] = ivn_src_idxs
                else:
                    ivn_src_idxs = self.base_idxs_to_ivn_src_idxs[base_idx]

            for ivn_src_idx in ivn_src_idxs:
                base = self.base_dataset[base_idx]
                ivn_src = self.base_dataset[ivn_src_idx]
                high_ivn = self.construct_high_intervention(base, ivn_src)
                low_base_input = self.construct_low_input(base)
                low_ivn_src = self.construct_low_input(ivn_src)
                weight = self.intervention_weight_fn(base, ivn_src) if \
                    self.intervention_weight_fn is not None else None
                yield {
                    "high_intervention": high_ivn,
                    "low_base_input": low_base_input,
                    "low_intervention_source": low_ivn_src,
                    "mapping": self.mapping,
                    "weight": weight,
                    "base_idx": base_idx,
                    "ivn_src_idx": ivn_src_idx.item()
                }

    def construct_high_intervention(self, base_ex, ivn_src_ex) -> antra.Intervention:
        """ Construct the high level intervention object given two examples.

        :param base_ex: Example retrieved from base dataset as base
        :param ivn_src_ex: Example retrieved from base dataset as intervention source
        """
        raise NotImplementedError

    def construct_low_input(self, example) -> antra.GraphInput:
        """ Construct a GraphInput object for the low level model"""
        raise NotImplementedError

    def collate_fn(self, batch):
        return cf_collate_fn(batch, self.mapping, batch_dim=self.batch_dim)

    def get_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)


# TODO: move this to separate utils module
def graph_input_collate_fn(batch, batch_dim=0):
    """Collate list of individual graph inputs into one."""
    ex = batch[0]
    key_leaves, non_batch_leaves = ex.key_leaves, ex.non_batch_leaves

    gi_dict = pack_graph_inputs(
        batch,
        batch_dim=batch_dim,
        non_batch_inputs=non_batch_leaves
    )

    gi = antra.GraphInput.batched(
        gi_dict,
        batch_dim=batch_dim,
        cache_results=False,
        key_leaves=key_leaves,
        non_batch_leaves=non_batch_leaves
    )

    return gi


def cf_collate_fn(batch, mapping, batch_dim=0):
    """Collate list of individual interventions into one. Similar to the one
    used in interchange dataset"""
    high_base = batch[0]["high_intervention"].base
    high_key_leaves, high_non_batch_leaves = high_base.key_leaves, high_base.non_batch_leaves
    low_base = batch[0]["low_base_input"]
    low_key_leaves, low_non_batch_leaves = low_base.key_leaves, low_base.non_batch_leaves

    high_base_dict, high_ivn_dict, high_loc_dict = pack_interventions(
        [d["high_intervention"] for d in batch],
        batch_dim=batch_dim,
        non_batch_inputs=high_non_batch_leaves
    )
    high_base_input = antra.GraphInput.batched(
        high_base_dict,
        batch_dim=batch_dim,
        cache_results=False,
        key_leaves=high_key_leaves,
        non_batch_leaves=high_non_batch_leaves
    )
    high_ivn = antra.Intervention.batched(
        high_base_input, high_ivn_dict, high_loc_dict,
        batch_dim=batch_dim,
        cache_results=False
    )

    low_base_gi_dict = pack_graph_inputs(
        [d["low_base_input"] for d in batch],
        batch_dim=batch_dim,
        non_batch_inputs=low_non_batch_leaves
    )

    low_base_input = antra.GraphInput.batched(
        low_base_gi_dict,
        batch_dim=batch_dim,
        cache_results=False,
        key_leaves=low_key_leaves,
        non_batch_leaves=low_non_batch_leaves
    )

    low_ivn_src_gi_dict = pack_graph_inputs(
        [d["low_intervention_source"] for d in batch],
        batch_dim=batch_dim,
        non_batch_inputs=low_non_batch_leaves
    )

    low_ivn_src_input = antra.GraphInput.batched(
        low_ivn_src_gi_dict,
        batch_dim=batch_dim,
        cache_results=False,
        key_leaves=low_key_leaves,
        non_batch_leaves=low_non_batch_leaves
    )

    weights = None
    if batch[0]["weight"] is not None:
        weights = torch.tensor([d["weight"] for d in batch])
    base_idxs = [d["base_idx"] for d in batch]
    ivn_src_idxs = [d["ivn_src_idx"] for d in batch]
    return {
        "high_intervention": high_ivn,
        "low_base_input": low_base_input,
        "low_intervention_source": low_ivn_src_input,
        "weights": weights,
        "base_idxs": base_idxs,
        "ivn_src_idxs": ivn_src_idxs,
        "mapping": mapping
    }