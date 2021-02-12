import torch
from compgraph import GraphInput, GraphNode, ComputationGraph

# TODO: specify list of modules
# TODO: `with` wrapper to define modules on the go


class CompGraphConstructor:
    """Automatically construct a `ComputationGraph` from a `torch.nn.Module`.

    Currently, the constructor will automatically treat the submodules of
    type `torch.nn.Module` of the provided module as nodes in the computation
    graph.

    The constructor only works if the outputs of every submodule directly feeds
    into another one, without any intermediate steps outside the scope of a
    submodule's forward() function.
    """

    def __init__(self, module, submodules=None):
        """ Initialize constructor from a pytorch module

        :param module: `torch.nn.Module` the module in question.
        :param submodules: see CompGraphConstructor.construct().
        """
        assert isinstance(module, torch.nn.Module), \
            "Must provide an instance of a nn.Module"

        self.module = module

        self.name_to_node = {}
        self.module_to_name = {}
        self.current_input = None

        submodules = module.named_children() if not submodules \
            else submodules
        if isinstance(submodules, dict):
            submodules = submodules.items()

        for name, submodule in submodules:
            # construct nodes based on children modules of module
            # no edges yet, they are created during construct()
            node = GraphNode(name=name, forward=submodule.forward)
            self.name_to_node[name] = node
            self.module_to_name[submodule] = name

            submodule.register_forward_pre_hook(self.pre_hook)
            submodule.register_forward_hook(self.post_hook)

    @classmethod
    def construct(cls, module, *args, device=None, submodules=None):
        """ Construct a computation graph given a torch.nn.Module

        We must provide a sample input to the torch.nn.Module to construct
        the computation graph. The intermediate output values of each node will
        be automatically stored in the nodes of the graph that is constructed.

        :param module: `torch.nn.Module`
        :param args: inputs as-is to `module.forward()`
        :param device: optional, `torch.Device`, move inputs to this device
        :param submodules: optional, `dict(str -> torch.nn.Module)` or
            `list_like(tuple(str, torch.nn.Module)).
            Specifies submodules that form the nodes in the computation graph.
            Each submodule is associated with a unique string name.
        :return: (ComputationGraph, GraphInput) where `g` is the constructed
            computation graph, `input_obj` is a GraphInput object based on args,
            which is required for further intervention experiments on the
            ComputationGraph.
        """
        constructor = cls(module, submodules=submodules)
        g, input_obj = constructor.make_graph(*args, device=device)
        return g, input_obj

    def pre_hook(self, module, inputs):
        """ Executed before the module's forward() and unpacks inputs

        We track how modules are connected by augmenting the outputs of a
        module with the its own name, which can be read when the outputs become
        the inputs of another module.

        :param module: torch.nn.Module, submodule of self.module
        :param inputs: tuple of (Tensor, str) tuples, Tensors are inputs to
             1module.forward()`, str denotes name of module that outputted each
             input
        :return: modified input that is actually passed to module.forward()
        """

        name = self.module_to_name[module]
        current_node = self.name_to_node[name]
        print("I am in module", self.module_to_name[module],
              "I have %d inputs" % len(inputs))

        if not all(isinstance(x, tuple) and len(x) == 2 for x in inputs):
            raise RuntimeError("At least one input to \"%s\" is not an output "
                               "of a named module!" % name)

        actual_inputs = tuple(t[0] for t in inputs)

        # get information about which modules do the inputs come from
        if any(x[1] is None for x in inputs):
            if not all(x[1] is None for x in inputs):
                raise NotImplementedError("Nodes currently don't support mixed "
                                          "leaf and non-leaf inputs")
            current_node.children = []
            if self.current_input is not None:
                raise NotImplementedError(
                    "Currently only supports one input leaf!")
            else:
                self.current_input = GraphInput({name: actual_inputs})
        else:
            current_node.children = [self.name_to_node[t[1]] for t in inputs]

        return actual_inputs

    def post_hook(self, module, inputs, output):
        """Executed after module.forward(), repackages outputs into tuple format

        :param module: torch.nn.Module, submodule of self.Module
        :param inputs: the inputs to module.forward()
        :param output: outputs from module.forward()
        :return: modified output of module, and may be passed on to next module
        """
        name = self.module_to_name[module]
        current_node = self.name_to_node[name]

        # store output info in cache
        current_node.base_cache[self.current_input] = output

        # package node name together with output
        return output, name

    def make_graph(self, *args, device=None):
        """ construct a computation graph given a nn.Module """
        if device:
            inputs = tuple((x.to(device), None) for x in args)
        else:
            inputs = tuple((x, None) for x in args)
        print("inputs", inputs)
        res, root_name = self.module(*inputs)
        graph_input_obj = self.current_input
        self.current_input = None
        root = self.name_to_node[root_name]
        return ComputationGraph(root), graph_input_obj
