# antra

Lightweight package for defining computation graphs and performing intervention experiments

## Table of Contents

   * [antra](#antra)
      * [Table of Contents](#table-of-contents)
      * [Installation and dependencies](#installation-and-dependencies)
      * [Basic Usage](#basic-usage)
      * [Defining a computation graph](#defining-a-computation-graph)
      * [Basic computation](#basic-computation)
      * [Interventions](#interventions)
         * [Setting up an intervention](#setting-up-an-intervention)
         * [Computing the intervention](#computing-the-intervention)
      * [Interventions on specific indices and slices of vectors/tensors](#interventions-on-specific-indices-and-slices-of-vectorstensors)
         * [Specifying intervention location as a string](#specifying-intervention-location-as-a-string)
         * [Specifying intervention location using the LOC object](#specifying-intervention-location-using-the-loc-object)
         * [Computing interventions with specified indexing location](#computing-interventions-with-specified-indexing-location)
         * [How this works under the hood](#how-this-works-under-the-hood)
      * [Batched computation and intervention](#batched-computation-and-intervention)
      * [Value caching and keys](#value-caching-and-keys)
      * [Caching control](#caching-control)
      * [Abstracted computation graphs](#abstracted-computation-graphs)


## Installation and dependencies

Install `antra` using `pip`:

```
$ pip install antra
```

`antra` is implemented using vanilla Python 3, and its basic usage doesn't depend on other packages.

To utilize `antra`'s batch operations, please [install `pytorch`](https://pytorch.org/get-started/locally/)

## Basic Usage

`antra`'s main functionality is to perform efficient interventions on computation processes, which is
essential for the causal analysis of a program or algorithm (see [section below](#interventions) on interventions).

Using `antra`, users can declaratively construct a computation graph that implements a computation process of interest. 
The computation graph contains nodes, which can either be a *leaf*, which serve as the input points of the graph, 
or can represent a function, which takes in the output values of other nodes as inputs and returns a  value. 
The computation graph must be a directed acyclic graph, and must have one single *root* node which outputs the final result of 
the entire graph.

Each node in the computation graph contains an internal `dict` that caches 
the result for different inputs, making it efficient to access intermediate values in the graph and intervene on the 
computation graph, at the expense of extra memory space.

`antra` is lightweight and flexible. It is agnostic to the input and output types as well as the content
of each node's functions. Optionally, if you have `pytorch` installed, `antra` can perform computations 
and interventions in batches, which is useful for analyzing numerically intensive systems such as neural networks.

Note that `antra`'s primary purpose is to provide a lightweight scaffolding to convert an algorithm/program/computation process
into a computation graph and perform interventions on it. It does not perform back-propogation on its own.

## Defining a computation graph

To define a computation graph, first define the nodes in it using `antra.GraphNode` by specifying each node's `name`,
and for non-leaf nodes, its function (called its `forward` function) and its children nodes, who provide input values
to the function's arguments.

After defining the nodes, pass in the root node to the `antra.ComputationGraph` constructor, to construct the 
computation graph.

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
Note that the `GraphNode` constructor takes in an arbitrary number of positional arguments that serve as its
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

## Basic computation

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

## Interventions

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
`node2 = [-18, -38, -58]` and `root = -114`. This will be the "base" input.

Suppose we want to ask what would happen if we set the value of `node1` to `[-20, -20, -20]` during the computation, 
ignoring the result of `x * y`.

This would be an intervention: when we compute `node2`, we set `node1 = [-20, -20, -20]` and the input value
`y = [2, 2, 2]` , to get `node2 = [22, 22, 22]` and subsequently `root = 66`.

### Setting up an intervention
To perform the above intervention using `antra`, we construct a `antra.Intervention` object. `antra` allows many 
alternative ways to do this.

An intervention object requires a base input and the intervention itself, both of which being a mapping from (leaf) node names 
to numeric values. This mapping can be given either using a `dict` or a `GraphInput` object that wraps around it.

```python
import torch
from antra import GraphInput, Intervention

base_input_dict = {"x": torch.tensor([10, 20, 30]), "y": torch.tensor([2, 2, 2])}
base_input_gi_obj = GraphInput(base_input_dict)
intervention_dict = {"node1": torch.tensor([-20, -20, -20])}
intervention_gi_obj = GraphInput(intervention_dict)

# the following are equivalent:

interv = Intervention(base_input_gi_obj, intervention_dict)
interv = Intervention(base_input_gi_obj, intervention_gi_obj)
interv = Intervention(base_input_dict, intervention_dict)
interv = Intervention(base_input_dict, intervention_gi_obj)
```
Another alternative is to specify the intervened nodes and values separately using the `set_intervention()` method,
which makes code more readable, instead of packing everything into the constructor function.
```python
interv1 = Intervention(base_input_gi_obj)
interv1.set_intervention("node1", torch.tensor([-20, -20, -20]))
```

### Computing the intervention

Use the `g.intervene()` method to performing the intervention on the graph. It returns two root outputs, computed
without and with the intervention respectively.
```python
base_result, interv_result = g.intervene(interv)
print(base_result, interv_result) # -63, 66
```
To retrieving intermediate node values during the intervention, use `g.compute_node()`, which takes in a node name and 
an `Intervention` object. You can also retrieve the intermediate node value computed without the intervention by passing
in the `GraphInput` object of the intervention's base input.
```python
node2_interv_value = g.compute_node("node2", interv)
print(node2_interv_value) # tensor([22,22,22])

# the following are equivalent
node2_base_value = g.compute_node("node2", base_input_gi_obj) # base_input_gi_obj defined above
node2_base_value = g.compute_node("node2", interv.base) 
print(node2_base_value) # tensor([20, 40, 60])
```

Note that `antra` can detect which nodes in the computation graph are affected by the intervention, so that it does not
compute parts of the graph whose results are going to be unused (such as `x * y` in the above case) due to the intervention.


## Interventions on specific indices and slices of vectors/tensors

You can intervene on only specific elements or slices of vectors and tensors using `antra` by specifying an array index as you
would normally do using square brackets `[]`.

Using the same example as above, suppose we would like to set only the zeroth and first element of `node1` to `-10` and `-20` respectively, and keep the
third element as it is, i.e. 
```
node1 = x * y // element-wise multiplication
node1[:2] = torch.tensor([-10, -20]) // intervention
node2 = -1 * node1 + y
root = node2.sum(dim=-1)
```

`antra` supports various ways to do this.

### Specifying intervention location as a string

When constructing the intervention object, you can add the bracket notation as usual but in the form of a string appended
after the node name:
```python
intervention_dict = {"node1[:2]": torch.tensor([-10, -20])}
interv = Intervention(base_input_gi_obj, intervention_dict) # base input as defined above
# --- or ---
interv = Intervention(base_input_gi_obj)
interv.set_intervention("node1[:2]", torch.tensor([-10, -20]))
```
This may be less flexible when you'd like to dynamically modify the indexing and slicing in the brackets, which brings us to
the next method:

### Specifying intervention location using the `LOC` object

Normally, we can access slices of a `pytorch` tensor and `numpy` array using square brackets, e.g. `a[:,0]` gets us the
0th element in each row of the tensor `a`. But can we somehow store the indexing information `[:,0]` into a variable,
say `idx`, such that `a[idx]` is equivalent as `a[:,0]`? This would be useful as now we can flexibly pass `idx` around 
and use it on different tensors/arrays.

`antra` provides a helper object, `LOC` (as in `LOCation`) that does the above, as seen in this example:
```python
from antra import LOC
idx = LOC[:2]
torch.all(a[idx] == a[:2]) # True 
```
We can then use the `LOC` object to provide the indexing information for an intervention, by specifying a `LOC` object 
for each node. There are two alternative ways to do this:
```python
intervention_dict = {"node1": torch.tensor([-10, -20])}
location_dict = {"node1": LOC[:2]}
interv = Intervention(base_input_gi_obj, intervention_dict, location_dict) # base input as defined above
# --- or ---
interv = Intervention(base_input_gi_obj)
interv.set_intervention("node1", torch.tensor([-10, -20]))
interv.set_location("node1", LOC[:2])
```

### Computing interventions with specified indexing location

This is the same as described in the [section above](#computing-the-intervention).

### How this works under the hood
> The bracket notation `[]` on a python object is essentially a call to `__getitem__()`
> (to retrieve values) or `__setitem__()` (for value assignments) builtin methods. Within the brackets,
> comma `,` denote a `tuple`, and colons `:` are a shorthand for python `slice` objects. 
> 
> For example, for a pytorch tensor `x`, the notation `:,:2` within `x[:,:2]` essentially represents `tuple(slice(None, None, None), slice(None, 2, None))`. 
> `torch.tensor`'s `__getitem__()` and `__setitem__()` take this as its argument and interpret it as accessing
> the first two elements of each row.
> 
> And finally, the `LOC` object is just a dummy object with a `__getitem__()` function that directly returns whatever is 
> passed into the brackets:
```python
# antra/location.py
class Location:
    def __getitem__(self, item):
        return item
```

## Batched computation and intervention


## Value caching and keys

## Caching control

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




## Abstracted computation graphs

