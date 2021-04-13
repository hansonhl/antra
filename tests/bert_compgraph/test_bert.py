import pytest
import torch
from torch.utils.data import DataLoader
import transformers
from antra import *
from antra.compgraphs.bert import *

from .dataset import SentimentDataHelper

@pytest.fixture()
def bert_tokenizer():
    return transformers.BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture()
def sentiment_data(bert_tokenizer):
    return SentimentDataHelper("tests/bert_compgraph/data/senti.train.tsv",
                               "tests/bert_compgraph/data/senti.dev.tsv",
                               "tests/bert_compgraph/data/senti.test.tsv",
                               tokenizer=bert_tokenizer)

@pytest.fixture()
def bert_model():
    print("loading model...")
    return transformers.BertModel.from_pretrained("bert-base-uncased")

@pytest.fixture()
def bert_masked_lm_model():
    print("loading model...")
    return transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")

@pytest.fixture()
def bert_masked_seq_classification_model():
    print("loading model...")
    return transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased")

def test_bert_forward(sentiment_data, bert_model):
    model = bert_model

    device = torch.device("cuda")
    model = model.to(device)
    g = BertModelCompGraph(model)
    # print("leaves", g.leaves)

    dataloader = DataLoader(sentiment_data.dev, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # print("input_ids.shape", batch["input_ids"].shape)
            # print("attn_mask.shape", batch["attention_mask"].shape)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                return_dict=True
            )
            pooler_output = outputs.pooler_output
            # print("batch from dataloader", batch)
            gi = BertGraphInput(batch)
            res = g.compute(gi)
            assert torch.allclose(res, pooler_output)


def test_bert_intervention(sentiment_data, bert_model):
    model = bert_model

    device = torch.device("cuda")
    model = model.to(device)
    g = BertModelCompGraph(model)

    dataloader = DataLoader(sentiment_data.dev, batch_size=32, shuffle=True)
    with torch.no_grad():
        for _ in range(3):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size, _ = batch["input_ids"].shape

                gi = BertGraphInput(batch)
                base_res = g.compute(gi)
                layer_11_output = g.compute_node("bert_layer_11", gi).detach().clone()

                interv_values = torch.randn(batch_size, 50)
                interv_dict = {"bert_layer_11[:,0,:50]": interv_values}
                interv = Intervention.batched(gi, interv_dict)
                interv_before, interv_after = g.intervene(interv)
                assert torch.allclose(interv_before, base_res)

                layer_11_output[:,0,:50] = interv_values

                pooler_input = layer_11_output[:,0]
                dense_out = model.pooler.dense(pooler_input)
                expected_after = model.pooler.activation(dense_out)

                assert torch.allclose(interv_after, expected_after, atol=1e-06, rtol=1e-04)
                # for interv_res, expected_res in zip(interv_after, expected_after):
                #     assert torch.allclose(interv_res, expected_res)


def test_bert_masked_lm_model_intervention(sentiment_data, bert_masked_lm_model):
    model = bert_masked_lm_model

    device = torch.device("cuda")
    model = model.to(device)
    g = BertForMaskedLMCompGraph(model)

    dataloader = DataLoader(sentiment_data.dev, batch_size=32, shuffle=True)
    with torch.no_grad():
        for _ in range(3):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size, _ = batch["input_ids"].shape

                gi = BertGraphInput(batch)
                base_res = g.compute(gi)
                layer_11_output = g.compute_node("bert_layer_11", gi).detach().clone()

                interv_values = torch.randn(batch_size, 50)
                interv_dict = {"bert_layer_11[:,0,:50]": interv_values}
                interv = Intervention.batched(gi, interv_dict, cache_results=False)
                interv_before, interv_after = g.intervene(interv)
                assert torch.allclose(interv_before, base_res)

                layer_11_output[:,0,:50] = interv_values

                expected_after = g.lm_head_module(layer_11_output)

                assert torch.allclose(interv_after, expected_after, atol=1e-04, rtol=1e-04)
                # for interv_res, expected_res in zip(interv_after, expected_after):
                #     assert torch.allclose(interv_res, expected_res)

def test_bert_seq_classification_intervention(sentiment_data, bert_masked_seq_classification_model):
    model = bert_masked_seq_classification_model
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    g = BertForSequenceClassificationCompGraph(model)

    dataloader = DataLoader(sentiment_data.dev, batch_size=32, shuffle=True)
    with torch.no_grad():
        for _ in range(3):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size, _ = batch["input_ids"].shape

                gi = BertGraphInput(batch)
                base_res = g.compute(gi)
                layer_11_output = g.compute_node("bert_layer_11", gi).detach().clone()

                interv_values = torch.randn(batch_size, 50)
                interv_dict = {"bert_layer_11[:,0,:50]": interv_values}
                interv = Intervention.batched(gi, interv_dict, cache_results=False)
                interv_before, interv_after = g.intervene(interv)
                assert torch.allclose(interv_before, base_res)

                layer_11_output[:,0,:50] = interv_values
                pooler_output = g.bert_model.pooler(layer_11_output)
                expected_after = g.cls(pooler_output)

                assert torch.allclose(interv_after, expected_after, atol=1e-06, rtol=1e-04)
                # for interv_res, expected_res in zip(interv_after, expected_after):
                #     assert torch.allclose(interv_res, expected_res)