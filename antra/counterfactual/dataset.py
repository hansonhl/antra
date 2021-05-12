import torch
from torch.utils.data import Dataset, IterableDataset

import antra
from antra.interchange.batched import pack_interventions
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

        return {
            "high_intervention": high_ivn,
            "weight": weight
        }

    def construct_intervention(self, base, ivn_source) -> antra.Intervention:
        """ Construct the high level intervention object given two examples.

        :param base:
        :param ivn_source:
        """
        raise NotImplementedError

    def collate_fn(self, batch):
        return collate_fn(batch, self.mapping, batch_dim=self.batch_dim)

class RandomCounterfactualDataset(IterableDataset):
    def __init__(
            self,
            base_dataset: Dataset,
            mapping: AbstractionMapping,
            intervention_weight_fn: Optional[Callable] = None,
            batch_dim: int = 0,
            num_random_bases=50000,
            num_random_ivn_srcs=20,
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

    def __iter__(self):
        rand_base_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_bases]
        for base_idx in rand_base_idxs:
            ivn_src_idxs = torch.randperm(self.base_dataset_len)[:self.num_random_ivn_srcs]
            for ivn_src_idx in ivn_src_idxs:
                base = self.base_dataset[base_idx.item()]
                ivn_src = self.base_dataset[ivn_src_idx.item()]
                high_ivn = self.construct_intervention(base, ivn_src)
                weight = self.intervention_weight_fn(base, ivn_src) if \
                    self.intervention_weight_fn is not None else None
                yield {
                    "high_intervention": high_ivn,
                    "mapping": self.mapping,
                    "weight": weight
                }

    def construct_intervention(self, base, ivn_source) -> antra.Intervention:
        """ Construct the high level intervention object given two examples.

        :param base:
        :param ivn_source:
        """
        raise NotImplementedError

    def collate_fn(self, batch):
        return collate_fn(batch, self.mapping, batch_dim=self.batch_dim)



def collate_fn(batch, mapping, batch_dim=0):
    """Collate list of individual interventions into one. Similar to the one
    used in interchange dataset"""
    high_base = batch[0]["high_intervention"].base
    high_key_leaves, high_non_batch_leaves = high_base.key_leaves, high_base.non_batch_leaves

    high_base_dict, high_ivn_dict, high_loc_dict = pack_interventions(
        [d["high_intervention"] for d in batch],
        batch_dim=batch_dim,
        non_batch_inputs=high_non_batch_leaves
    )
    high_base_input = antra.GraphInput(
        high_base_dict, batched=True, batch_dim=batch_dim,
        cache_results=True,
        key_leaves=high_key_leaves,
        non_batch_leaves=high_non_batch_leaves
    )
    high_ivn = antra.Intervention.batched(
        high_base_input, high_ivn_dict, high_loc_dict,
        batch_dim=batch_dim,
        cache_results=True
    )
    weights = None
    if batch[0]["weight"] is not None:
        weights = torch.tensor([d["weight"] for d in batch])
    return {
        "high_intervention": high_ivn,
        "weights": weights,
        "mapping": mapping
    }