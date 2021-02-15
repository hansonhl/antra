# compgraph

Lightweight package for defining computation graphs and performing intervention experiments

## Installation and dependencies

Install `compgraph` using `pip`:

```
$ pip install compgraph
```

`compgraph` is implemented using vanilla Python 3, and basic usage doesn't depend on other packages.

To utilize `compgraph`'s batch operations, please [install `pytorch`](https://pytorch.org/get-started/locally/)

## Basic usage

`compgraph` supports defining a computation graph composed of computation nodes. Each node is either a *leaf*, which 
serve as the input of the graph, or can be a function, which takes in the output values of other nodes as inputs, and 
returns a single value. The computation graph must be a directed acyclic graph, and must have one single *root* node
which outputs the final result of the entire graph.

Each node contains an internal dictionary that caches the result for different inputs, so that one can efficiently 
retrieve intermediate values of the graph and perform interventions on the computation graph, at the expense
of some memory space.

### Defining a computation graph

To define a computation graph, first define the nodes in it using `compgraph.GraphNode` by specifying each node's `name`,
and for non-leaf nodes, its function (called its `forward` function), and its children, which are the nodes whose outputs 
are the inputs of the forward function.

After defining the nodes, pass in the root node to the `compgraph.ComputationGraph` constructor.

The `compgraph` package is agnostic to the input and output types of each node's functions, unless working with 
batched computations.

For instance, the following defines the nodes in the computation graph for the series of equations 

```
node1 = x + y
node2 = -1 * node1
root = node2.sum(dim=-1)
```

where x, y are `pytorch` tensors.

```python
from compgraph import GraphNode, ComputationGraph

x = GraphNode.leaf("x")
y = GraphNode.leaf("y")

node1_f = lambda x, y: x + y
node1 = GraphNode(x, y, name="node1", forward=node1_f)

node2_f = lambda z: -1 * z
node2 = GraphNode(node1, name="node2", forward=node2_f)

root_f = lambda r: r.sum(dim=-1)
root = GraphNode(node2, name="root", forward=root_f)

g = ComputationGraph(root)
```

The `GraphNode` constructor takes in an arbitrary number of positional arguments that serve as its
children. **Note that the ordering of the node's children must be same as defined in the function**.

Alternatively, as syntactic sugar, one can define computation graph nodes using decorators on functions:

```python
from compgraph import GraphNode, ComputationGraph

x = GraphNode.leaf("x")
y = GraphNode.leaf("y")

@GraphNode(x, y)
def node1(x, y):
    return x + y

@GraphNode(node1)
def node2(z):
    return -1 * z

@GraphNode(node2)
def root(x):
    return x.sum(dim=-1)

print(node1) # GraphNode('node1')
print(node1.children) # [GraphNode("x"), GraphNode("y")]

g = ComputationGraph(root)
```

The above is equivalent to the previous method. The decorator `@GraphNode()` takes in an arbitrary number of child nodes 
as the node's arguments, and will automatically treat the function that it decorates as node's `forward` function. Same
as the previous method, *the ordering of the children in the decorator arguments must be same as defined in the function*.
Finally, the node's name will be the string that is same as function's name.

The variable names for the function's parameters can be different from the child node names that appear in the decorator.

### Basic computation

Having defined the computation graph, one can run computations with it, by specifying the inputs to the graph using a
`compgraph.GraphInput` object, and using the graph's `compute()` method. The `compute_node()` method computes the output
value of a specific node in the graph.

```python
import torch
from compgraph import GraphInput
# ...... g is the computation graph that is defined above

input_dict = {"x": torch.tensor([10, 20, 30]), "y": torch.tensor([1, 1, 1])}
in1 = GraphInput(input_dict)
res = g.compute(in1)
print(res)  # -63

node1_res = g.compute_node("node1", in1)
print(node1_res)  # tensor([11, 21, 31])
```

Because each node in the graph automatically caches its result on each input during `compute()`, calling `compute_node()`
will retrieve the cached return value the key of the input (see section on value caching and keys below) and immediately 
return it, without performing any extra computation.

Note that directly calling `compute_node()` on an intermediate node may only run the computation graph partially, and
leave remaining downstream nodes uncomputed.

### Interventions


### Value caching and keys


### Caching control

Prevent a computation from caching a result on a certain input:

```python
in1 = GraphInput(input_dict, cache_results=False)
```

To prevent a node in the graph from caching results in general:

```python
node1_f = lambda x, y: x + y
node1 = GraphNode(x, y, name="node1", forward=node1_f, cache_results=False)

# --- or ---

@GraphNode(x, y, cache_results=False)
def node1(x, y):
    return x + y

# --- or ---
node1.cache_results = False
```

It is recommended to do this for nodes whose output values you are not interested in obtaining, or nodes that will not
be intervened on.


### Batched computation and intervention


### Abstracted computation graphs

