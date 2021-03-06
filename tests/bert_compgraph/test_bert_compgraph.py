import pytest
import torch
from torch.utils.data import DataLoader
import transformers
from antra.compgraphs.bert import BertCompGraph, BertGraphInput

from .dataset import SentimentDataHelper

@pytest.fixture()
def bert_tokenizer():
    return transformers.BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture()
def sentiment_data(bert_tokenizer):
    return SentimentDataHelper("data/senti.train.tsv",
                               "data/senti.dev.tsv",
                               "data/senti.test.tsv",
                               tokenizer=bert_tokenizer)


def test_bert_compgraph(sentiment_data):
    model = transformers.BertModel.from_pretrained("bert-base-uncased")

    device = torch.device("cuda")
    model = model.to(device)
    g = BertCompGraph(model)

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
                return_dict=True)
            pooler_output = outputs.pooler_output
            gi = BertGraphInput(batch)
            res = g.compute(gi)
            assert torch.all(torch.isclose(res, pooler_output))
