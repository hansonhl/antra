import pytest
import torch
from antra import GraphInput
from antra.compgraphs.bert import BertGraphInput


def test_to_cuda():
    d = {str(x): torch.randn(10) for x in range(5)}
    i = GraphInput(d)

    device = torch.device("cuda")
    i = i.to(device)

    assert all(t.is_cuda for t in i._values.values())

def test_to_cuda2():
    d = {
        "input_ids": torch.randint(10, (8, 12)),
        "attention_mask": torch.ones(8, 12),
        "token_type_ids": torch.ones(8, 12),
        "output_attentions": False
    }

    bgi = BertGraphInput(d)

    device = torch.device("cuda")
    bgic = bgi.to(device)

    for k, v in bgic.values.items():
        if isinstance(v, torch.Tensor):
            assert v.is_cuda
            temp_v = v.to(torch.device("cpu"))
            assert torch.allclose(temp_v, bgi[k])
        else:
            assert v == bgi[k]

    for i in range(len(bgic.keys)):
        assert bgic.keys[i] == bgi.keys[i]

    assert bgic.batch_dim == bgi.batch_dim
    assert bgic.key_leaves == bgi.key_leaves
    assert bgic.non_batch_leaves == bgi.non_batch_leaves


def test_immutability():
    d = {str(x): torch.randn(10) for x in range(5)}
    i = GraphInput(d)

    with pytest.raises(RuntimeError) as excinfo:
        i.values = {"a": torch.tensor([1,2,3])}
        assert "immutable" in excinfo
