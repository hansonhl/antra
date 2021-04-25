
import torch
import logging
from torch.nn.functional import softmax, sigmoid
from antra import ComputationGraph, GraphNode, GraphInput
from antra.utils import serialize

def generate_MLP_compgraph(MLP, activation, output_function = lambda x:x):
    """
    Converts an sklearn MLP Classifier to a comp graph.
    """
    input_leaf = GraphNode.leaf("input")
    curr_node = GraphNode(input_leaf,name ="hidden_layer_1" ,forward=lambda x: activation(torch.matmul(x,torch.from_numpy(MLP.coefs_[0])) + torch.from_numpy(MLP.intercepts_[0])))
    for i in range(1, MLP.n_layers_-1):
        name = "hidden_layer_" + str(i+1)
        f = lambda x: activation(torch.matmul(x,torch.from_numpy(MLP.coefs_[i])) + torch.from_numpy(MLP.intercepts_[i]))
        curr_node2 = GraphNode(curr_node,name=name,forward=f)
        curr_node = curr_node2
    if MLP.out_activation_ == "identity":
        output = GraphNode(curr_node,name="output",forward=lambda x: x)
    elif MLP.out_activation_ == "logistic":
        output = GraphNode(curr_node,name="output",forward=sigmoid)
    elif MLP.out_activation_ == "softmax":
        output = GraphNode(curr_node,name="outputt",forward=lambda x: softmax(x,dim=0))
    root = GraphNode(output,name="root",forward=output_function)
    return ComputationGraph(root)
