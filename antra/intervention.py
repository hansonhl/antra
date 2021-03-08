import re

from .location import Location
from .graph_input import GraphInput

import logging

logger = logging.getLogger(__name__)

from typing import Dict, Union, Sequence

class Intervention:
    """ A hashable intervention object """

    def __init__(self, base: Union[Dict, GraphInput],
                 intervention: Union[Dict, GraphInput]=None,
                 location: Dict=None,
                 cache_results: bool=True, cache_base_results:bool=True,
                 batched: bool=False, batch_dim: int=0, keys=None):
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
            If `base` is a GraphInput object then its `cache_results`
        :param batched: If true then indicates `values` contains a batch of
            inputs, i.e. the value of the dict must be a sequence.
        :param batch_dim: If inputs are batched and are pytorch tensors, the
            dimension for the batch
        :param keys: A unique key/hash value for each input value in the batch
        """
        intervention = {} if intervention is None else intervention
        location = {} if location is None else location
        # if batched and not (isinstance(base, GraphInput) and base.batched):
        #     raise ValueError("Must provide a batched GraphInput for a batched Intervention")

        self.cache_results = cache_results
        self.cache_base_results = cache_base_results
        self.affected_nodes = None
        self.batched = batched
        self.batch_dim = batch_dim

        self._setup(base, intervention, location)


    def _get_keys(self):
        # assume at this point base and intervention are already converted into
        # GraphInput objects, which already have
        assert isinstance(self.base, GraphInput)
        assert isinstance(self.intervention, GraphInput)

        if not self.batched:
            loc_key = tuple(sorted(
                (k, Location.loc_to_str(v)) for k, v in self.location.items()))
            return (self.base.keys, self.intervention.keys, loc_key)
        else:
            if self.intervention.is_empty():
                return self.base.keys
            else:
                loc_key = tuple(sorted((k, Location.loc_to_str(v)) for k, v in self.location.items()))
                base_key, interv_key = self.base.keys, self.intervention.keys
                if len(base_key) != len(interv_key):
                    raise RuntimeError("Must provide the same number of inputs in batch for base and intervention")
                return [(b, i, loc_key) for b, i in zip(self.base.keys, self.intervention.keys)]


    @classmethod
    def batched(cls, base: Union[Dict, GraphInput],
                intervention: Union[Dict, GraphInput]=None,
                location: Dict=None, cache_results: bool=True,
                cache_base_results: bool=True,
                batch_dim: int=0):
        """ Specify a batched intervention object"""
        return cls(base=base, intervention=intervention, location=location,
                   cache_results=cache_results, cache_base_results=cache_base_results,
                   batched=True, batch_dim=batch_dim)

    def _setup(self, base=None, intervention=None, location=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(
                    base, cache_results=self.cache_base_results,
                    batched=self.batched, batch_dim=self.batch_dim
                )
            self._base = base

        if location is not None:
            # specifying any value that is not None will overwrite self._locs
            location = {name: Location.process(loc) for name, loc in
                        location.items()}
        else:
            location = self._location

        if intervention is not None:
            if isinstance(intervention, GraphInput):
                intervention = intervention.values

            # extract indexing in names
            loc_pattern = re.compile(r"\[.*]")
            to_delete = []
            to_add = {}
            for name, value in intervention.items():
                # parse any index-like expressions in name
                loc_search = loc_pattern.search(name)
                if loc_search:
                    true_name = name.split("[")[0]
                    loc_str = loc_search.group().strip("[]")
                    loc = Location.parse_str(loc_str)
                    to_delete.append(name)
                    to_add[true_name] = value
                    location[true_name] = loc

            # remove indexing part in names
            for name in to_delete:
                intervention.pop(name)
            intervention.update(to_add)

            self._intervention = GraphInput(
                intervention, batched=self.batched, batch_dim=self.batch_dim
            )

        self._location = location
        if self.location and not self.cache_base_results or not self.base.cache_results:
            logger.warning(f"To intervene on a indexed location, results of the base run must be cached, "
                           f"but self.cache_base_results is set to false. Automatically converting it to True")
            self.cache_base_results = True
            self.base.cache_results = True

        for loc_name in self.location:
            if loc_name not in self.intervention:
                logger.warning(f"{loc_name} is given a location but does not have an intervention value")

        # setup keys
        self.keys = self._get_keys()

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
