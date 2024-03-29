
import torch
import logging
from torch.nn.functional import softmax, sigmoid
from antra import ComputationGraph, GraphNode, GraphInput
from antra.utils import serialize

def _generate_MLP_layer_function(activation, MLP, i):
    """ Generate a function for a layer in bert.
    Each layer function takes in a previous layer's hidden states and outputs,
    and only returns the hidden states.

    :param layer_module:
    :param i:
    :return: Callable function that corresponds to a bert layer
    """
    def _MLP_layer_function(x):
        return activation(torch.matmul(x,torch.from_numpy(MLP.coefs_[i])) + torch.from_numpy(MLP.intercepts_[i]))
    return _MLP_layer_function

def generate_MLP_compgraph(MLP, activation, output_function = lambda x:x):
    """
    Converts an sklearn MLP Classifier to a comp graph.
    """
    input_leaf = GraphNode.leaf("input")
    curr_node = GraphNode(input_leaf,name ="hidden_layer_1" ,forward=lambda x: activation(torch.matmul(x,torch.from_numpy(MLP.coefs_[0])) + torch.from_numpy(MLP.intercepts_[0])))
    for i in range(1,len(MLP.coefs_)):
        name = "hidden_layer_" + str(i+1)
        function = _generate_MLP_layer_function(activation, MLP, i)
        curr_node = GraphNode(curr_node,name=name,forward=function)
    if MLP.out_activation_ == "identity":
        output = GraphNode(curr_node,name="output",forward=lambda x: x)
    elif MLP.out_activation_ == "logistic":
        output = GraphNode(curr_node,name="output",forward=sigmoid)
    elif MLP.out_activation_ == "softmax":
        output = GraphNode(curr_node,name="output",forward=lambda x: softmax(x,dim=0))
    root = GraphNode(output,name="root",forward=output_function)
    return ComputationGraph(root)
