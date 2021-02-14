import re

from .location import Location
from .graph_input import GraphInput

from typing import Dict, Union, Sequence

class Intervention:
    """ A hashable intervention object """

    def __init__(self, base: Union[Dict, GraphInput],
                 intervention: Union[Dict, GraphInput]=None,
                 location: Dict=None,
                 cache_results: bool=True, cache_base_results:bool=False,
                 batched: bool=False, batch_dim: int=0, keys: Sequence=None):
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
        if batched and not (isinstance(base, GraphInput) and base.batched):
            raise ValueError("Must provide a batched GraphInput for a batched Intervention")

        self._setup(base, intervention, location)
        self.cache_results = cache_results
        self.cache_base_results = cache_base_results
        self.affected_nodes = None
        self.batched = batched
        self.batch_dim = batch_dim
        self.keys = keys
        if batched and not self.keys:
            raise ValueError("Must provide keys for each element of the batch!")

    @classmethod
    def batched(cls, base: Union[Dict, GraphInput], keys: Sequence,
                intervention: Union[Dict, GraphInput]=None,
                location: Dict=None, cache_results: bool=False,
                cache_base_results: bool=False,
                batch_dim: int=0):
        return cls(base=base, intervention=intervention, location=location,
                   cache_results=cache_results, cache_base_results=cache_base_results,
                   batched=True, batch_dim=batch_dim, keys=keys)

    def _setup(self, base=None, intervention=None, location=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(base, cache_results=self.cache_base_results)
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

            self._intervention = GraphInput(intervention)

        self._location = location
        for loc_name in self.location:
            if loc_name not in self.intervention:
                raise RuntimeWarning(
                    " %s in locs does not have an intervention value")

    @property
    def base(self):
        return self._base

    @property
    def intervention(self):
        return self._intervention

    @intervention.setter
    def intervention(self, values):
        self._setup(intervention=values, location={})

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, values):
        self._setup(location=values)

    def set_intervention(self, name, value):
        d = self._intervention.values if self._intervention is not None else {}
        d[name] = value
        self._setup(intervention=d,
                    location=None)  # do not overwrite existing locations

    def set_location(self, name, value):
        d = self._location if self._location is not None else {}
        d[name] = value
        self._setup(location=d)

    def __getitem__(self, name):
        return self._intervention.values[name]

    def __setitem__(self, name, value):
        self.set_intervention(name, value)


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
