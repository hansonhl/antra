import torch
import logging

from antra import ComputationGraph, GraphNode, GraphInput
from antra.utils import serialize

logger = logging.getLogger(__name__)

def _generate_bert_layer_fxn(layer_module, i):
    """ Generate a function for a layer in bert.
    Each layer function takes in a previous layer's hidden states and outputs,
    and only returns the hidden states.

    :param layer_module:
    :param i:
    :return: Callable function that corresponds to a bert layer
    """
    def _bert_layer_fxn(hidden_states, input_dict):
        head_mask = input_dict["head_mask"]

        layer_head_mask = head_mask[i] if head_mask is not None else None
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=input_dict["extended_attention_mask"],
            head_mask=layer_head_mask,
            encoder_hidden_states=input_dict["encoder_hidden_states"],
            encoder_attention_mask=input_dict["encoder_attention_mask"],
            output_attentions=input_dict["output_attentions"],
        )
        hidden_states = layer_outputs[0]
        return hidden_states

    return _bert_layer_fxn


def generate_bert_compgraph(bert_model, final_node="pool"):
    """
    Generate the computation graph structure for the basic Huggingface BertModel.

    This is essentially a re-implementation of its `forward()` function in terms
    of GraphNodes, so we can access and intervene on the intermediate outputs
    of each layer.

    Currently it does not support

    :param bert_model:
    :return:
    """
    #
    # input_ids = GraphNode.default_leaf("input_ids")
    # attention_mask = GraphNode.default_leaf("attention_mask")
    # token_type_ids = GraphNode.default_leaf("token_type_ids")
    # position_ids = GraphNode.default_leaf("position_ids")
    # head_mask = GraphNode.default_leaf("head_mask")
    # inputs_embeds = GraphNode.default_leaf("inputs_embeds")
    # output_attentions = GraphNode.default_leaf("output_attentions")
    # output_hidden_states = GraphNode.default_leaf("output_hidden_states")
    #

    input_leaf = GraphNode.leaf("input")

    @GraphNode(input_leaf, cache_results=False)
    def input_preparation(input_dict):
        """
        Prepare inputs for Bert.

        This node corresponds to all the preparation before the call to
        self.embeddings in BertModel.forward(). We package everything into
        the same "input_dict" which other nodes can use directly.

        :param input_dict: input into BertModel.forward() in the form of a dict.
        :return: input_dict with administrative stuff filled in
        """
        input_dict = {k: v for k, v in input_dict.items()}

        for key in ["input_ids", "attention_mask", "token_type_ids",
                    "position_ids", "head_mask", "inputs_embeds",
                    "output_attentions", "output_hidden_states",
                    "return_dict", "encoder_hidden_states",
                    "encoder_attention_mask"]:
            if key not in input_dict: input_dict[key] = None

        for key in ["output_attentions", "output_hidden_states", "return_dict"]:
            if input_dict[key]:
                logger.warning(f"You set {key} to True, which `antra` cannot support at the moment. "
                               f"It will be set to False now.")
            input_dict[key] = False

        if input_dict["input_ids"] is not None and input_dict["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_dict["input_ids"] is not None:
            input_shape = input_dict["input_ids"].size()
            batch_size, seq_length = input_shape
            device = input_dict["input_ids"].device
        elif input_dict["inputs_embeds"] is not None:
            input_shape = input_dict["inputs_embeds"].size()[:-1]
            batch_size, seq_length = input_shape
            device = input_dict["inputs_embeds"].device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_dict["attention_mask"] is None:
            input_dict["attention_mask"] = torch.ones(((batch_size, seq_length)), device=device)
        if input_dict["token_type_ids"] is None:
            input_dict["token_type_ids"] = torch.zeros(input_shape, dtype=torch.long, device=device)

        input_dict["extended_attention_mask"] = bert_model.get_extended_attention_mask(input_dict["attention_mask"], input_shape, device)
        input_dict["head_mask"] = bert_model.get_head_mask(input_dict["head_mask"], bert_model.config.num_hidden_layers)

        return input_dict

    # Embedding Node
    @GraphNode(input_preparation)
    def embed(input_dict):
        return bert_model.embeddings(
            input_ids=input_dict["input_ids"],
            position_ids=input_dict["position_ids"],
            token_type_ids=input_dict["token_type_ids"],
            inputs_embeds=input_dict["inputs_embeds"]
        )

    hidden_layer = embed

    # Bert Layers
    for i in range(len(bert_model.encoder.layer)):
        f = _generate_bert_layer_fxn(bert_model.encoder.layer[i], i)
        hidden_layer = GraphNode(hidden_layer, input_preparation,
                                name=f"bert_layer_{i}",
                                forward=f)

    # Output pooling, if specified
    if bert_model.pooler is not None and final_node == "pool":
        @GraphNode(hidden_layer)
        def pool(h):
            return bert_model.pooler(h)
        return pool

    elif final_node == "pool":
        raise ValueError("Final node cannot be pool because the given BERT model does not have a final pool layer!")

    elif final_node == "hidden":
        return hidden_layer
    else:
        raise ValueError(f"Invalid final node specification: {final_node}!")

class BertGraphInput(GraphInput):
    def __init__(self, input_dict, cache_results=True):
        keys = [serialize(x) for x in input_dict["input_ids"]]
        super().__init__(
            values={"input": input_dict},
            cache_results=cache_results,
            batched=True,
            batch_dim=0,
            keys=keys
        )

class BertModelCompGraph(ComputationGraph):
    def __init__(self, bert_model, final_node="pool"):
        self.bert_model = bert_model
        root = generate_bert_compgraph(self.bert_model, final_node=final_node)
        super().__init__(root)


class BertLMHeadModelCompGraph(ComputationGraph):
    def __init__(self, model):
        self.model = model
        self.bert_model = model.bert
        self.lm_head_module = model.cls

        assert self.bert_model.pooler is None
        bert_hidden = generate_bert_compgraph(self.bert_model, final_node="hidden")

        @GraphNode(bert_hidden, cache_results=False)
        def lm_head(h):
            return self.lm_head_module(h)

        super().__init__(lm_head)


# same as above
class BertForMaskedLMCompGraph(ComputationGraph):
    def __init__(self, model):
        self.model = model
        self.bert_model = model.bert
        self.lm_head_module = model.cls

        assert self.bert_model.pooler is None
        bert_hidden = generate_bert_compgraph(self.bert_model, final_node="hidden")

        @GraphNode(bert_hidden, cache_results=False)
        def lm_head(h):
            return self.lm_head_module(h)

        super().__init__(lm_head)


class BertForNextSentecePredictionCompGraph(ComputationGraph):
    def __init__(self, model):
        self.model = model
        self.bert_model = model.bert
        self.cls = model.cls
        assert self.bert_model.pooler is not None
        pooler = generate_bert_compgraph(self.bert_model, final_node="pool")

        @GraphNode(pooler)
        def cls_head(h):
            return self.cls(h)

        super().__init__(cls_head)


class BertForSequenceClassificationCompGraph(ComputationGraph):
    def __init__(self, model):
        self.model = model
        self.bert_model = model.bert
        self.dropout = model.dropout
        self.cls = model.classifier

        assert self.bert_model.pooler is not None
        pooler = generate_bert_compgraph(self.bert_model, final_node="pool")

        @GraphNode(pooler)
        def cls_head(h):
            return self.cls(self.dropout(h))

        super().__init__(cls_head)

class BertForTokenClassificationCompGraph(ComputationGraph):
    def __init__(self, model):
        self.model = model
        self.bert_model = model.bert
        self.dropout = model.dropout
        self.cls = model.classifier

        assert self.bert_model.pooler is None
        bert_hidden = generate_bert_compgraph(self.bert_model, final_node="hidden")

        @GraphNode(bert_hidden)
        def cls_head(h):
            return self.cls(self.dropout(h))

        super().__init__(cls_head)
