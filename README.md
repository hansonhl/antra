# antra

Lightweight package for defining computation graphs and performing intervention experiments

Table of Contents
=================

   * [antra](#antra)
      * [Installation and dependencies](#installation-and-dependencies)
      * [Basic Usage](#basic-usage)
         * [Defining a computation graph](#defining-a-computation-graph)
         * [Basic computation](#basic-computation)
         * [Interventions](#interventions)
         * [Value caching and keys](#value-caching-and-keys)
         * [Caching control](#caching-control)
      * [Batched computation and intervention](#batched-computation-and-intervention)
      * [Abstracted computation graphs](#abstracted-computation-graphs)

## Installation and dependencies

Install `antra` using `pip`:

```
$ pip install antra
```

`antra` is implemented using vanilla Python 3, and its basic usage doesn't depend on other packages.

To utilize `antra`'s batch operations, please [install `pytorch`](https://pytorch.org/get-started/locally/)

## Basic Usage

`antra` supports defining a computation graph composed of computation nodes. Each node is either a *leaf*, which 
serve as the input of the graph, or can be a function, which takes in the output values of other nodes as inputs, and 
returns a single value. The computation graph must be a directed acyclic graph, and must have one single *root* node
which outputs the final result of the entire graph.

Each node contains an internal dictionary that caches the result for different inputs, so that one can efficiently 
retrieve intermediate values of the graph and perform interventions (see section below on interventions) 
on the computation graph, at the expense of extra memory space.

### Defining a computation graph

To define a computation graph, first define the nodes in it using `antra.GraphNode` by specifying each node's `name`,
and for non-leaf nodes, its function (called its `forward` function) and its children nodes, who provide input values
to the function's arguments.

After defining the nodes, pass in the root node to the `antra.ComputationGraph` constructor, to construct the 
computation graph.

`antra` is agnostic to the input and output types of each node's functions, unless working with 
batched computations and interventions, which currently requires `pytorch`.

In the following we use an example to explain how to construct a computation graph using `antra`.

For instance, suppose we have the following graph that takes in two vectors `x` and `y` as inputs:

![example computation graph](example_compgraph.png)

Which can be expressed in equations as:

```
node1 = x * y   // element-wise product
node2 = -1 * node1 + y
root = node2.sum(dim=-1)
```

We can define the graph using `antra` as:

```python
from antra import GraphNode, ComputationGraph

x = GraphNode.leaf("x")
y = GraphNode.leaf("y")

node1_f = lambda x, y: x * y
node1 = GraphNode(x, y, name="node1", forward=node1_f)

node2_f = lambda z, y: -1 * z + y
node2 = GraphNode(node1, y, name="node2", forward=node2_f)

root_f = lambda r: r.sum(dim=-1)
root = GraphNode(node2, name="root", forward=root_f)

g = ComputationGraph(root)
```

The `GraphNode` constructor takes in an arbitrary number of positional arguments that serve as its
children. **Note that the ordering of the node's children must be same as defined in the function**.

Alternatively, as syntactic sugar, one can define computation graph nodes using decorators on functions:

```python
from antra import GraphNode, ComputationGraph

x = GraphNode.leaf("x")
y = GraphNode.leaf("y")

@GraphNode(x, y)
def node1(x, y):
    return x * y

@GraphNode(node1, y)
def node2(z, y):
    return -1 * z + y

@GraphNode(node2)
def root(x):
    return x.sum(dim=-1)

print(node1) # GraphNode('node1')
print(node1.children) # [GraphNode("x"), GraphNode("y")]

g = ComputationGraph(root)
```

The above is equivalent to the previous method. The decorator `@GraphNode()` takes in an arbitrary number of `GraphNode`
objects that are the current node's children, and will make the function that it decorates as the `forward` function. 
Similar to the previous method, **the ordering of the children in the `@GraphNode()` must match the order of arguments
in the function**. Finally, the decorator will take the function name to be the node's name, and the function now becomes
a `GraphNode` object that can be used in the remainder of the code.

Note that the *variable names* of the function's arguments can be different from the child node names 
that appear in the decorator.

### Basic computation

Having defined the computation graph, one can run computations with it by first specifying the inputs to the graph using a
`antra.GraphInput` object, which provides the values of each leaf node in the graph.
Then one can use the graph's `compute()` method to obtain the output value at the root node. 
The `compute_node()` method computes the output value of a specific node in the graph.

```python
import torch
from antra import GraphInput
# ...... g is the computation graph that is defined above

input_dict = {"x": torch.tensor([10, 20, 30]), "y": torch.tensor([2, 2, 2])}
in1 = GraphInput(input_dict)
res = g.compute(in1)
print(res)  # -114

node1_res = g.compute_node("node1", in1)
print(node1_res)  # tensor([20, 40, 60])
```

Because each node in the graph automatically caches its result on each input during `compute()`, calling `compute_node()`
will retrieve the cached return value by looking up the input's key in an internal `dict` 
(see section on value caching and keys below) and immediately return it, without performing any extra computation.

Note that directly calling `compute_node()` on an intermediate node may only run the computation graph partially, and
leave remaining downstream nodes uncomputed.

### Interventions

The `antra` package supports *interventions* on the computation graph. An intervention on a computation graph is 
essentially setting the output values of one or more intermediate nodes in the graph and computing the rest of the  
nodes as usual, but with these altered intermediate values. Intervention experiments can be useful for 
inferring the causal behavior of models. 

An intervention requires a "base" input which provides the input values for all the leaf nodes, and values for 
intervening on the intermediate nodes.

For instance, using the same computation graph as above:
```
node1 = x * y     // element-wise product
node2 = -1 * node1 + y
root = node2.sum(dim=-1)
```
Suppose we first run the computation graph with inputs `x = [10, 20, 30], y = [2, 2, 2]` and get `node1 = [20, 40, 60]`, 
`node2 = [-18, -38, -58]` and `root = -114`. 

Suppose we want to ask what would happen if we set the value of `node1` to `[-20, -20, -20]` during the computation, 
ignoring the result of `x * y`.

This would be an intervention: when we compute `node2`, we set `node1 = [-20, -20, -20]` and the input value
`y = [2, 2, 2]` , to get `node2 = [22, 22, 22]` and subsequently `root = 66`.

To perform the above intervention using `antra`, we define a `antra.Intervention` object.
It requires a `GraphInput` object as its "base" input, and another `GraphInput` specifying the intervention values.
These `GraphInput` objects can be substituted with a `dict` mapping from node names to values.

`antra` can detect which nodes in the computation graph are affected by the intervention, so that it does not
compute parts of the graph whose results are going to be unused (such as `x * y` in the above case) due to the intervention.

```python
import torch
from antra import GraphInput, Intervention

input_dict = {"x": torch.tensor([10, 20, 30]), "y": torch.tensor([2, 2, 2])}
in1 = GraphInput(input_dict)
intervention_dict = {"node1": torch.tensor([-20, -20, -20])}
in2 = GraphInput(intervention_dict)

# the following are equivalent:

interv1 = Intervention(in1, intervention_dict)
interv1 = Intervention(in1, in2)
interv1 = Intervention(input_dict, intervention_dict)
interv1 = Intervention(input_dict, in2)
```

Performing the intervention on the graph, and retrieving intermediate node values during the intervention:

```python
base_result, interv_result = g.intervene(interv1)
print(base_result, interv_result) # -63, 66

node2_interv_value = g.compute_node("node2", interv1)
print(node2_interv_value) # tensor([22,22,22])
```

### Value caching and keys


### Caching control

**Prevent a computation from caching a result on a certain input**

```python
in1 = GraphInput(input_dict, cache_results=False)
_ = g.compute(in1)   # intermediate values won't be cached
```

**Prevent a node in the graph from caching results in general**

```python
node1_f = lambda x, y: x + y
node1 = GraphNode(x, y, name="node1", forward=node1_f, cache_results=False)

# --- or ---W

@GraphNode(x, y, cache_results=False)
def node1(x, y):
    return x + y

# --- or ---
node1.cache_results = False
```
It is recommended to do this for nodes whose output values you are not interested in obtaining, or nodes that will not
be intervened on.

**Prevent caching results during an intervention**

```python
interv1 = Intervention(input_dict, intervention_dict, cache_results=False, cache_base_results=False)
_, _ = g.intervene(interv1)
```


## Batched computation and intervention


## Abstracted computation graphs

