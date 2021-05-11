import re
import pprint
from collections import defaultdict

from .location import Location, location_to_str, LocationType, deserialize_location
from .graph_input import GraphInput
from .utils import serialize, serialize_batch
from .realization import Realization

import logging

logger = logging.getLogger(__name__)

from typing import *

class Intervention:
    """ A hashable intervention object """

    def __init__(
        self,
        base: Union[Dict, GraphInput],
        intervention: Union[Dict, GraphInput] = None,
        location: Dict = None,
        cache_results: bool = True,
        cache_base_results: bool = True,
        batched: bool = False,
        batch_dim: int = 0,
        realization: Optional[Union[List[Realization], Realization]] = None
    ):
        """ Construct an intervention experiment.

        :param base: `GraphInput` or `dict(str->Any)` containing the "base"
            input to a graph, where we intervene on the intermediate outputs of
            this input instance.
        :param intervention: `GraphInput` or `dict(str->Any)` denoting the node
            names and corresponding values that we want to set the nodes
        :param location: `dict(str->str or int or Loc or tuple)` optional,
            indices to intervene on part of a tensor or array
        :param cache_results: If true then cache the results of this
            intervention during computation.
        :param cache_base_results: Only useful if `base` is provided as a dict.
            If true then cache the results of the base run during computation.
            If `base` is a GraphInput object then its `cache_results` will
            override this setting.
        :param batched: If true then indicates `values` contains a batch of
            inputs, i.e. the value of the dict must be a sequence.
        :param batch_dim: If inputs are batched and are pytorch tensors, the
            dimension for the batch
        :param realization: Reference of the Realization object that contains
            info about where the values of this intervention came from
        """
        intervention = {} if intervention is None else intervention
        location = {} if location is None else location
        # if batched and not (isinstance(base, GraphInput) and base.batched):
        #     raise ValueError("Must provide a batched GraphInput for a batched Intervention")

        self.cache_results = cache_results
        self.cache_base_results = cache_base_results
        self.affected_nodes = None
        self.multi_loc_nodes = set()
        self.batched = batched
        self.batch_dim = batch_dim
        self.realization = realization

        self._setup(base, intervention, location)

    @classmethod
    def from_realization(
        cls,
        base: Union[Dict, GraphInput],
        realization: Realization,
        cache_results: bool = True,
        cache_base_results: bool = True
    ):
        ivn_dict, loc_dict = defaultdict(list), defaultdict(list)

        for (node_name, ser_low_loc), val in realization.items():
            ivn_dict[node_name].append(val)
            low_loc = deserialize_location(ser_low_loc)
            loc_dict[node_name].append(low_loc)

        for node_name in ivn_dict.keys():
            if len(ivn_dict[node_name]) == 1:
                ivn_dict[node_name] = ivn_dict[node_name][0]
            if len(loc_dict[node_name]) == 1:
                if loc_dict[node_name][0] is None:
                    del loc_dict[node_name]
                else:
                    loc_dict[node_name] = loc_dict[node_name][0]

        ivn_dict = dict(ivn_dict)
        loc_dict = dict(loc_dict)
        return cls(base, intervention=ivn_dict, location=loc_dict, batched=False,
                   cache_results=cache_results, cache_base_results=cache_base_results,
                   realization=realization)

    @classmethod
    def batched(cls, base: Union[Dict, GraphInput],
                intervention: Union[Dict, GraphInput] = None,
                location: Dict[str, LocationType] = None,
                cache_results: bool = True,
                cache_base_results: bool = True,
                batch_dim: int = 0,
                realization: Optional[List[Realization]] = None):
        """ Specify a batched intervention object"""
        return cls(base=base, intervention=intervention, location=location,
                   cache_results=cache_results, cache_base_results=cache_base_results,
                   batched=True, batch_dim=batch_dim, realization=realization)

    def _setup(self, base=None, intervention=None, location=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(
                    base, cache_results=self.cache_base_results,
                    batched=self.batched, batch_dim=self.batch_dim
                )
            self._base = base

        if location is not None:
            # specifying any value that is not None, including empty dict {}, will overwrite self._location
            new_location = {}
            self.multi_loc_nodes = set()
            for node_name, loc in location.items():
                if isinstance(loc, list):
                    new_location[node_name] = [Location.process(l) for l in loc]
                    self.multi_loc_nodes.add(node_name)
                else:
                    new_location[node_name] = Location.process(loc)
            location = new_location
        else:
            # do not overwrite self._location
            location = self._location

        if intervention is not None:
            if isinstance(intervention, GraphInput):
                intervention = intervention.values

            # extract indexing in names
            loc_pattern = re.compile(r"\[.*]")
            to_delete = []
            to_add = {}

            for full_name, value in intervention.items():
                # parse any index-like expressions in name
                loc_search = loc_pattern.search(full_name)
                if loc_search:
                    node_name = full_name.split("[")[0]  # bare node name without indexing
                    loc_str = loc_search.group().strip("[]")
                    loc = Location.parse_str(loc_str)
                    to_delete.append(full_name)

                    if node_name in intervention:
                        to_add[node_name] = intervention[node_name]

                    if node_name not in to_add:
                        to_add[node_name] = value
                    else:
                        # use list of values for different locations in same node
                        prev_values = to_add[node_name] if isinstance(to_add[node_name], list) else [to_add[node_name]]
                        prev_values.append(value)
                        to_add[node_name] = prev_values
                        self.multi_loc_nodes.add(node_name)

                    if node_name not in location:
                        location[node_name] = loc
                    else:
                        # different locations in same node
                        prev_locs = location[node_name] if isinstance(location[node_name], list) else [
                            location[node_name]]
                        prev_locs.append(loc)
                        location[node_name] = prev_locs
                        self.multi_loc_nodes.add(node_name)

            # remove indexing part in names
            for node_name in to_delete:
                intervention.pop(node_name)
            intervention.update(to_add)

            if self.multi_loc_nodes:
                self._validate_multi_loc(intervention, location, self.multi_loc_nodes)

            interv_keys = self._get_interv_graphinput_keys(intervention, location, self.multi_loc_nodes)

            self._intervention = GraphInput(
                intervention, batched=self.batched, batch_dim=self.batch_dim,
                keys=interv_keys
            )

        self._location = location

        # if self.location and not self.cache_base_results or not self.base.cache_results:
        #     logger.warning(f"To intervene on a indexed location, results of the base run must be cached, "
        #                    f"but self.cache_base_results is set to false. Automatically converting it to True")
        #     self.cache_base_results = True
        #     self.base.cache_results = True

        for loc_name in self.location:
            if loc_name not in self.intervention:
                logger.warning(f"{loc_name} is given a location but does not have an intervention value")

        # setup keys
        self.keys = self._get_keys()

    def _validate_multi_loc(self, intervention, location, multi_loc_nodes):
        # validate multi-location nodes
        for node_name in multi_loc_nodes:
            if not isinstance(location[node_name], list) or \
                not isinstance(intervention[node_name], list) or \
                len(location[node_name]) != len(intervention[node_name]):
                raise ValueError(
                    f"Mismatch between number of locations and values for node {node_name}")

    def _get_interv_graphinput_keys(self, intervention, location, multi_loc_nodes):
        dict_for_keys = {}
        for node_name in intervention:
            if node_name not in multi_loc_nodes:
                k = node_name
                if node_name in location:
                    k += location_to_str(location[node_name], add_brackets=True)
                dict_for_keys[k] = intervention[node_name]
            else:
                for loc, value in zip(location[node_name], intervention[node_name]):
                    k = node_name + location_to_str(loc, add_brackets=True)
                    dict_for_keys[k] = value
        if self.batched:
            return serialize_batch(dict_for_keys, dim=self.batch_dim)
        else:
            return serialize(dict_for_keys)

    def _get_keys(self):
        # assume at this point base and intervention are already converted into
        # GraphInput objects, which should have keys
        assert isinstance(self.base, GraphInput)
        assert isinstance(self.intervention, GraphInput)

        if self.intervention.is_empty():
            return self.base.keys

        if not self.batched:
            return (self.base.keys, self.intervention.keys)
        else:
            base_key, interv_key = self.base.keys, self.intervention.keys
            if len(base_key) != len(interv_key):
                raise RuntimeError("Must provide the same number of inputs in batch for base and intervention")
            return [(b, i) for b, i in zip(self.base.keys, self.intervention.keys)]

    @property
    def base(self):
        return self._base

    @property
    def intervention(self):
        return self._intervention

    # @intervention.setter
    # def intervention(self, values):
    #     self._setup(intervention=values, location={})
    #     if not self.batched:
    #         self.keys = self._prepare_keys()

    @property
    def location(self):
        return self._location

    # @location.setter
    # def location(self, values):
    #     self._setup(location=values)
    #     if not self.batched:
    #         self.keys = self._prepare_keys()

    def set_intervention(self, name, value):
        d = self._intervention.values.copy() if self._intervention is not None else {}
        d[name] = value
        self._setup(intervention=d, location=None)  # do not overwrite existing locations

    def set_location(self, name, value):
        d = self._location if self._location is not None else {}
        d[name] = value
        self._setup(location=d)

    def __getitem__(self, name):
        return self._intervention.values[name]

    # def __setitem__(self, name, value):
    #     self.set_intervention(name, value)

    def find_affected_nodes(self, graph):
        """Find nodes affected by this intervention in a computation graph.

        Stores its results by setting the `affected_nodes` attribute of the
        `Intervention` object.

        :param graph: `ComputationGraph` object
        :return: python `set` of nodes affected by this experiment
        """
        if self.affected_nodes is not None:
            return self.affected_nodes

        if self.intervention is None or len(self.intervention) == 0:
            self.affected_nodes = set()
            return set()

        affected_nodes = set()

        def affected(node):
            # check if children are affected, use simple DFS
            is_affected = False
            for c in node.children:
                if affected(c):  # we do not want short-circuiting here
                    affected_nodes.add(node.name)
                    is_affected = True
            if node.name in self.intervention:
                affected_nodes.add(node.name)
                is_affected = True
            return is_affected

        affected(graph.root)
        self.affected_nodes = affected_nodes
        return affected_nodes

    def get_batch_size(self):
        if not self.batched: return None
        return self.base.get_batch_size()

    def is_empty(self):
        return self.intervention.is_empty()

    def to(self, device):
        """Move all data to a pytorch Device.

        This does NOT modify the original GraphInput object but returns a new
        one. """
        # assert all(isinstance(t, torch.Tensor) for _, t in self._values.items())
        new_base = self.base.to(device)
        new_intervention = self.intervention.to(device)
        new_locs = self.location.copy()

        return Intervention(new_base, new_intervention, new_locs,
                            cache_results=self.cache_results,
                            cache_base_results=self.cache_base_results,
                            batched=self.batched,
                            batch_dim=self.batch_dim)

    def __repr__(self, shorten=True):
        ivn = None
        # shorten outputs when bert layers are involved
        if shorten:
            try:
                if 'bert' in self.intervention.keys[0][0]:
                    ivn = f'{self.intervention.keys[0][0]} ... (hidden omitted)'
            except:
                pass
            try:
                if 'bert' in self.intervention.keys[0][0][0]:
                    ivn = f'{self.intervention.keys[0][0][0]} ... (hidden omitted)'
            except:
                pass
        # otherwise, we just do the default
        if ivn is None:
            ivn = self.intervention.keys

        repr_dict = {
            "base": self.base.keys,
            "interv": ivn,
            "locs": self.location
        }
        return pprint.pformat(repr_dict, indent=1, compact=True)
