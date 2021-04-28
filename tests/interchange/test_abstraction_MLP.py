#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import PremackDataset
from equality_experiment import EqualityExperiment
import os
from pprint import pprint 
from antra import *
import utils
import torch
from torch.nn import ReLU
from antra.interchange.batched import BatchedInterchange
from antra.location import location_to_str, reduce_dim
from antra.compgraphs.MLP import generate_MLP_compgraph
from antra.location import generate_all_locations
from torch.nn.functional import sigmoid


# In[ ]:



utils.fix_random_seeds()


# In[ ]:


hidden_dim = 8
embed_size = 1
params = dict(
    embed_dims=[embed_size],
    hidden_dims=[hidden_dim],
    alphas=[0.001],
    learning_rates=[0.01],
    n_trials=1
)


# In[ ]:


experiment_h2 = EqualityExperiment(
    dataset_class=PremackDataset, 
    n_hidden=4,
    train_sizes=[600000],
    **params)

df_h2, MLP,training_data = experiment_h2.run()


# In[ ]:


print(df_h2)


# In[ ]:


relu = ReLU(inplace=False)
comp_graph = generate_MLP_compgraph(MLP,relu , output_function = lambda x: x> 0.5)


# In[ ]:


low_model = comp_graph


# In[ ]:


def equals(a,b):
    results = []
    for j in range(len(a)):
        x = a[j]
        y = b[j]
        results.append(torch.equal(x,y))
    return torch.tensor(results)

class PremackProgram(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")
        leaf4 = GraphNode.leaf("leaf4")

        @GraphNode(leaf1,leaf2)
        def node1(x,y):
            return equals(x,y)

        @GraphNode(leaf3,leaf4)
        def node2(x,y):
            return equals(x,y)

        @GraphNode(node1, node2)
        def root(w,v):
            return equals(w, v)

        super().__init__(root)

        
class PremackProgramA(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")
        leaf4 = GraphNode.leaf("leaf4")

        @GraphNode(leaf1,leaf2)
        def node1(x,y):
            return equals(x,y)

        @GraphNode(node1, leaf3,leaf4)
        def root(w,l3,l4):
            return equals(w, equals(l3,l4))

        super().__init__(root)

class PremackProgramB(ComputationGraph):
    def __init__(self):
        leaf1 = GraphNode.leaf("leaf1")
        leaf2 = GraphNode.leaf("leaf2")
        leaf3 = GraphNode.leaf("leaf3")
        leaf4 = GraphNode.leaf("leaf4")

        @GraphNode(leaf3,leaf4)
        def node2(x,y):
            return equals(x,y)

        @GraphNode(leaf1,leaf2,node2)
        def root(l1,l2,w):
            return equals(w, equals(l1,l2))

        super().__init__(root)


# In[ ]:





# In[ ]:


convert = lambda x: x 
data_size = 30
for i in range(data_size):
    print(training_data[42+i])



modelname = "AB"
if modelname == "A":
    high_model = PremackProgramA()
if modelname == "B":
    high_model = PremackProgramB()
if modelname == "AB":
    high_model = PremackProgram()
low_inputs = [
        GraphInput({
            "input": torch.from_numpy(x)
        }) for x in training_data[42:42+data_size]
    ]

high_inputs = [
        GraphInput({
            "leaf1": torch.tensor(convert(x[embed_size*0:embed_size*1])),
            "leaf2": torch.tensor(convert(x[embed_size*1:embed_size*2])),
            "leaf3": torch.tensor(convert(x[embed_size*2:embed_size*3])),
            "leaf4": torch.tensor(convert(x[embed_size*3:embed_size*4])),
        }) for x in training_data[42:42+data_size]
    ]

if modelname == "AB":
    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(convert(x[embed_size*0:embed_size*1])),
            "leaf2": torch.tensor(convert(x[embed_size*1:embed_size*2])),
            "leaf3": torch.tensor(convert(x[embed_size*2:embed_size*3])),
            "leaf4": torch.tensor(convert(x[embed_size*3:embed_size*4])),
        }, {
            "node1": torch.tensor(y),
            "node2": torch.tensor(z),            
        }) for x in training_data[42:42+data_size] for y in (True, False) for z in (True, False)
    ]
if modelname == "A":
    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(convert(x[embed_size*0:embed_size*1])),
            "leaf2": torch.tensor(convert(x[embed_size*1:embed_size*2])),
            "leaf3": torch.tensor(convert(x[embed_size*2:embed_size*3])),
            "leaf4": torch.tensor(convert(x[embed_size*3:embed_size*4])),
        }, {
            "node1": torch.tensor(y)        
        }) for x in training_data[42:42+data_size] for y in (True, False) 
    ]
if modelname == "B":
    high_ivns = [
        Intervention({
            "leaf1": torch.tensor(convert(x[embed_size*0:embed_size*1])),
            "leaf2": torch.tensor(convert(x[embed_size*1:embed_size*2])),
            "leaf3": torch.tensor(convert(x[embed_size*2:embed_size*3])),
            "leaf4": torch.tensor(convert(x[embed_size*3:embed_size*4])),
        }, {
            "node2": torch.tensor(y)        
        }) for x in training_data[42:42+data_size] for y in (True, False) 
    ]



# for i in range(data_size):
#     print(low_model.compute(low_inputs[i]), high_model.compute(high_inputs[i]), low_model.compute(low_inputs[i]) == high_model.compute(high_inputs[i]))
#     if low_model.compute(low_inputs[i]) != high_model.compute(high_inputs[i])[0]:
#         print("False", i)
#     assert model.predict(training_data[42:42+data_size])[i] == 0 and low_model.compute(low_inputs[i])[0] == False or model.predict(training_data[42:42+data_size])[i] == 1 and low_model.compute(low_inputs[i])[0] == True


# In[ ]:



fixed_node_mapping =  {x: {"input": LOC[embed_size*i:i+embed_size]} for i,x in enumerate(["leaf1",  "leaf2", "leaf3", "leaf4"])}
fixed_node_mapping["root"] = {"root": None}
low_nodes_to_indices = {
    'hidden_layer_1': [LOC[:,3:4],[LOC[:,0:3], LOC[:,4:7]], [LOC[:,0:2], LOC[:,3:]]],
#         "hidden_layer_1": generate_all_locations(hidden_dim) ,
#         "hidden_layer_2": generate_all_locations(hidden_dim) ,
#         "hidden_layer_3": generate_all_locations(hidden_dim) ,
#         "hidden_layer_4": generate_all_locations(hidden_dim) ,
    }

B = BatchedInterchange(
        low_model=low_model,
        high_model=high_model,
        low_inputs=low_inputs,
        high_inputs=high_inputs,
        high_interventions=high_ivns,
        low_nodes_to_indices=low_nodes_to_indices,
        fixed_node_mapping=fixed_node_mapping,
        store_low_interventions=True,
        result_format="simple",
        batch_size=16,
    )


# In[ ]:


# print(training_data[0].shape)
# print(len(model.coefs_))
# print(low_inputs[0]["input"].shape)
# print(torch.matmul(low_inputs[0]["input"], torch.from_numpy(model.coefs_[0])).shape)
# print(relu(torch.matmul(relu(torch.matmul(low_inputs[0]["input"], torch.from_numpy(model.coefs_[0])) + torch.from_numpy(model.intercepts_[0])), torch.from_numpy(model.coefs_[1])) + torch.from_numpy(model.intercepts_[1])).shape)
# print(torch.matmul(torch.matmul(torch.matmul(low_inputs[0]["input"], torch.from_numpy(model.coefs_[0])), torch.from_numpy(model.coefs_[1])),torch.from_numpy(model.coefs_[2])).shape)
# for i in range(len(model.coefs_)):
#     print(model.coefs_[i].shape)
# print(low_model.root.children[0].children[0].children[0])
# print(low_model.root.children[0].children[0].children[0].forward(torch.matmul(low_inputs[0]["input"], torch.from_numpy(model.coefs_[0]))))
# print(low_model.compute_node("input", low_inputs[0]))
# print(low_model.compute_node("hidden_layer_1", low_inputs[0]))
# print(low_model.compute_node("hidden_layer_2", low_inputs[0]))
# print(low_model.compute_node("hidden_layer_3", low_inputs[0]))
# print(low_model.compute_node("root", low_inputs[0]))
find_abstr_res = B.find_abstractions()


# In[ ]:


success_list = []
for result,mapping in find_abstr_res:
#     print(mapping)
    low_node1 = list(mapping["node1"].keys())[0]
    low_node2 = list(mapping["node2"].keys())[0]
    success = True
    success_count = 0
    total = 0
    for keys in result:
        low_ivn_key, high_ivn_key = keys
        low_ivn = B.low_keys_to_interventions[low_ivn_key]
        high_ivn = B.high_keys_to_interventions[high_ivn_key]

        _, low_res = low_model.intervene(low_ivn)
        _, high_res = high_model.intervene(high_ivn)
#         print("low_res", low_res)
#         print("high_res", high_res)
#         print("compare", high_res==low_res)
#         print("result", result[keys])
#         print("not result", not result[keys])
#         print("low intervention:",low_ivn.intervention.values)
#         print(f"low loc {low_node}", location.location_to_str(low_ivn.location[low_node], add_brackets=True))
#         print("lowbase:", low_ivn.base.values)
#         print("high intervetion:", high_ivn.intervention.values)
#         print("highbase:", high_ivn.base.values)
#         print("success:", result[keys])
        x = low_ivn.base["input"]
        N1I = None

        for i in range(0,len(MLP.coefs_)):
            x = relu(torch.matmul(x,torch.from_numpy(MLP.coefs_[i])) + torch.from_numpy(MLP.intercepts_[i]))
            if i ==0:
                ivn_value1 = low_ivn.intervention[low_node1]
                if isinstance(mapping["node1"][low_node1], tuple):
                    x[mapping["node1"][low_node1][1]] = ivn_value1[2]
                    N1I = ivn_value1[2]
                else:
                    x[mapping["node1"][low_node1][0][1]] = ivn_value1[0]
                    x[mapping["node1"][low_node1][1][1]] = ivn_value1[1]
                    N1I = (ivn_value1[0], ivn_value1[1])
                ivn_value2 = low_ivn.intervention[low_node2]                    
                if isinstance(mapping["node2"][low_node2], tuple):
                    x[mapping["node2"][low_node2][1]] = ivn_value2[2]
                    N1I = ivn_value1[2]                    
                else:
                    x[mapping["node2"][low_node2][0][1]] = ivn_value2[0]
                    x[mapping["node2"][low_node2][1][1]] = ivn_value2[1]
                    N2I = (ivn_value1[0], ivn_value1[1])   
#             print(str(i+1) + "th layer MLP",x)
#             print(str(i+1) + "th layer Compgraph",  low_model.compute_node("hidden_layer_" + str(i+1), low_ivn))
        expected_low_res = sigmoid(x) > 0.5
#         print("result MLP",expected_low_res)
#         print("result compgraph" ,  low_model.compute_node("output", low_ivn))
        A = equals(high_ivn.base["leaf1"], high_ivn.base["leaf2"])
        C =  not (equals(high_ivn.base["leaf3"], high_ivn.base["leaf4"])) 
        if "node1" in high_ivn.intervention:
            A = high_ivn.intervention["node1"]
        if "node2" in high_ivn.intervention:
            C = high_ivn.intervention["node2"]
        expected_high_res = A == C

#         print("node1", N1I, "node2", N2I, x, )
#         print(low_ivn)
#         print("low", expected_low_res, low_res)
#         print(high_ivn)
#         print("high", expected_high_res, high_res)
        assert expected_low_res == low_res
        assert expected_high_res == high_res
        assert (expected_low_res == expected_high_res) == result[keys]
        if False in result[keys]:
            success = False
        
        success_count += sum([1 if x else 0 for x in result[keys]])
        total += len(result[keys])
    success_list.append((success_count/total,success,mapping))
        
percents, bools, mappings = zip(*success_list)
maxp = max(percents)
print(maxp)
print(len(percents))
print("nay")
for percentage, success, mapping in success_list:
    if success:
        print("yay")
    if percentage == maxp:
        pprint(mapping)


# In[ ]:


pprint(success_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




