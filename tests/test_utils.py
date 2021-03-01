import pytest
import torch
import numpy as np

from antra.utils import serialize, serialize_batch
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
    s = serialize(x)
    d = deserialize(s)
    if isinstance(x == d, bool):
        assert x == d
    else:
        assert torch.all(x == d)

dataset = [([[1., 2.,], [3., 4.,], [5., 6.,]],
            [[[0, 1, 2], [10, 11, 12, ]],
             [[100, 101, 102, ], [110, 111, 112, ]],
             [[200, 201, 202, ], [210, 211, 212, ]]]
           ),
           ([3., 4., 5.], [3, 4, 5])]

@pytest.mark.parametrize("x, y", dataset)
def test_serialize_torch_batch_dim0(x, y):
    input1 = torch.tensor(x)
    input2 = torch.tensor(y)
    d = {"input1": input1, "input2": input2}
    res = serialize_batch(d, dim=0)
    assert len(res) == 3
    assert all(isinstance(ex, tuple) for ex in res)
    assert all(ex[0][0] == "input1" and ex[1][0] == "input2" for ex in res)
    for i, ex in enumerate(res):
        assert torch.allclose(torch.tensor(ex[0][1]), input1[i])
        assert torch.allclose(torch.tensor(ex[1][1]), input2[i])

@pytest.mark.parametrize("x, y", dataset)
def test_serialize_numpy_batch_dim0(x, y):
    input1 = np.array(x)
    input2 = np.array(y)
    d = {"input1": input1, "input2": input2}
    res = serialize_batch(d, dim=0)

    assert len(res) == 3
    assert all(isinstance(ex, tuple) for ex in res)
    assert all(ex[0][0] == "input1" and ex[1][0] == "input2" for ex in res)
    for i, ex in enumerate(res):
        assert np.allclose(np.array(ex[0][1]), input1[i])
        assert np.allclose(np.array(ex[1][1]), input2[i])


dataset = [([[1., 2.,], [3., 4.,], [5., 6.,]],
            [[[0, 1, 2], [10, 11, 12, ]],
             [[100, 101, 102, ], [110, 111, 112, ]],
             [[200, 201, 202, ], [210, 211, 212, ]]]
           )]
@pytest.mark.parametrize("x, y", dataset)
def test_serialize_torch_batch_dim1(x, y):
    input1 = torch.tensor(x)
    input2 = torch.tensor(y)
    d = {"input1": input1, "input2": input2}
    res = serialize_batch(d, dim=1)
    assert len(res) == 2
    assert all(isinstance(ex, tuple) for ex in res)
    assert all(ex[0][0] == "input1" and ex[1][0] == "input2" for ex in res)
    for i, ex in enumerate(res):
        assert torch.allclose(torch.tensor(ex[0][1]), input1[:,i])
        assert torch.allclose(torch.tensor(ex[1][1]), input2[:,i,:])

@pytest.mark.parametrize("x, y", dataset)
def test_serialize_numpy_batch_dim1(x, y):
    input1 = np.array(x)
    input2 = np.array(y)
    d = {"input1": input1, "input2": input2}
    res = serialize_batch(d, dim=1)

    assert len(res) == 2
    assert all(isinstance(ex, tuple) for ex in res)
    assert all(ex[0][0] == "input1" and ex[1][0] == "input2" for ex in res)
    for i, ex in enumerate(res):
        assert np.allclose(np.array(ex[0][1]), input1[:,i])
        assert np.allclose(np.array(ex[1][1]), input2[:,i,:])
