import pytest
import torch
from compgraph import GraphInput



def test_to_cuda():
    d = {str(x): torch.randn(10) for x in range(5)}
    i = GraphInput(d)

    device = torch.device("cuda")
    i = i.to(device)

    assert all(t.is_cuda for t in i.values.values())


def test_immutability():
    d = {str(x): torch.randn(10) for x in range(5)}
    i = GraphInput(d)

    with pytest.raises(RuntimeError) as excinfo:
        i.values = {"a": torch.tensor([1,2,3])}
        assert "immutable" in excinfo
