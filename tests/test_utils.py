import pytest
import torch
import numpy as np

from antra.utils import serialize, serialize_batch, idx_by_dim
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

t1 = torch.tensor([[3],[4],[5],[6]])
t2 = torch.tensor([1, 2, 3, 4])
t3 = torch.tensor([[0., 1.,], [10., 11.,], [20., 21.], [30., 31.]])
t4 = torch.stack((torch.zeros(3,3), torch.ones(3,3), 2 * torch.ones(3,3)), dim=0)

dataset = [
    (t1, 1, torch.tensor([4])),
    (t1, 3, torch.tensor([6])),
    (t2, 2, torch.tensor(3)),
    (t3, 0, torch.tensor([0., 1.])),
    (t3, 2, torch.tensor([20., 21.])),
    (t4, 1, torch.ones(3,3))
]

@pytest.mark.parametrize("x, idx, expected", dataset)
def test_idx_by_dim_0(x, idx, expected):
    got = idx_by_dim(x, idx, 0)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected)