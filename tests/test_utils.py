import pytest
import torch

from antra.utils import serialize
from antra.torch_utils import deserialize

dataset = [torch.tensor(1.),
           torch.tensor([1.]),
           torch.tensor([1., 2., 3., 4., 5.]),
           torch.tensor([[1., 2., 3., 4., 5.]]),
           torch.tensor([[1., 2.,], [3., 4.,], [5., 6.,]]),
           torch.tensor([[[0., 1., 2.], [10., 11., 12.,]],
                          [[100., 101., 102.,], [110., 111., 112.,]],
                          [[200., 201., 202.,], [210., 211., 212.,]]]),
           torch.tensor(1),
           torch.tensor([1]),
           torch.tensor([1, 2, 3, 4, 5]),
           torch.tensor([[1, 2, 3, 4, 5]]),
           torch.tensor([[1, 2, ], [3, 4, ], [5, 6, ]]),
           torch.tensor([[[0, 1, 2], [10, 11, 12, ]],
                         [[100, 101, 102, ], [110, 111, 112, ]],
                         [[200, 201, 202, ], [210, 211, 212, ]]])
           ]

@pytest.mark.parametrize("x", dataset)
def test_serialize(x):
    print(type(x))
    s = serialize(x)
    d = deserialize(s)
    if isinstance(x == d, bool):
        assert x == d
    else:
        assert torch.all(x == d)