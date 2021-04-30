import numpy as np
import random
from tests.interchange.utils import randvec



class EqualityDataset:

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, embed_dim=50, n_pos=500, n_neg=500, flatten=True):
        """Creates simple equality datasets, which are basically lists
        of `((vec1, vec2), label)` instances, where `label == POS_LABEL`
        if `vec1 == vec2`, else `label == NEG_LABEL`. With `flatten=True`,
        the instances become `(vec1;vec2, label)`.

        Parameters
        ----------
        embed_dim : int
            Sets the dimensionality of the individual component vectors.
        n_pos : int
        n_neg : int
        flatten : bool
            If False, instances are of the form ((vec1, vec2), label).
            If True, vec1 and vec2 are concatenated, creating instances
            (x, label) where len(x) == embed_dim*2.

        Usage
        -----
        dataset = EqualityDataset()
        X, y = dataset.create()

        Attributes
        ----------
        embed_dim : int
        n_pos : int
        n_neg : int
        flatten : bool

        """
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.flatten = flatten

    def create(self):
        """Main interface

        Attributes
        ----------
        data : list
            Shuffled version of the raw instances, ignoring `self.flatten`.
            Thus, these are all of the form `((vec1, vec2), label)`

        X : np.array
            The dimensionality depends on `self.flatten`. If it is
            False, then `X.shape == (n_pos+n_neg, 2, embed_dim)`. If it
            is True, then `X.shape == (n_pos+n_neg, embed_dim*2)`.

        y : list
            Containing `POS_LABEL` and `NEG_LABEL`. Length: n_pos+n_neg

        Returns
        -------
        self.X, self.y

        """
        self.data = []
        self.data += self._create_pos()
        self.data += self._create_neg()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten:
            data = [(np.concatenate(x), label) for x, label in data]
        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y
        return self.X, self.y

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for pair, label in self.data for x in pair}
        other_vecs = {tuple(x) for pair, label in other_dataset.data for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."

    def _create_pos(self):
        data = []
        for _ in range(self.n_pos):
            vec = randvec(self.embed_dim)
            rep = (vec, vec)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_neg(self):
        data = []
        for _ in range(self.n_neg):
            vec1 = randvec(self.embed_dim)
            vec2 = vec1.copy()
            while np.array_equal(vec1, vec2):
                vec2 = randvec(self.embed_dim)
            rep = (vec1, vec2)
            data.append((rep, self.NEG_LABEL))
        return data


class PremackDataset:

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, embed_dim=50, n_pos=500, n_neg=500,
                 flatten_root=True, flatten_leaves=True, intermediate=False):
        """Creates Premack datasets. Conceptually, the instances are

        (((a, b), (c, d)), label)

        where `label == POS_LABEL` if (a == b) == (c == d), else
        `label == NEG_LABEL`. With `flatten_leaves=True`, these become

        ((a;b, c;d), label)

        and with `flatten_root=True`, these become

        (a;b;c;d, label)

        and `flatten_root=True` means that `flatten_leaves=True`, since
        we can't flatten the roof without flattening the leaves.

        Parameters
        ----------
        embed_dim : int
            Sets the dimensionality of the individual component vectors.
        n_pos : int
        n_neg : int
        flatten_root : bool
        flatten_leaves : bool

        Usage
        -----
        dataset = EqualityDataset()
        X, y = dataset.create()

        Attributes
        ----------
        embed_dim : int
        n_pos : int
        n_neg : int
        flatten_root : bool
        flatten_leaves : bool
        n_same_same : n_pos / 2
        n_diff_diff : n_pos / 2
        n_same_diff : n_neg / 2
        n_diff_same : n_neg / 2

        Raises
        ------
        ValueError
            If `n_pos` or `n_neg` is not even, since this means we
            can't get an even distribtion of the two sub-types of
            each of those classes while also staying faithful to
            user's expected number of examples for each class.

        """
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.n_same_same = int(n_pos / 2)
        self.n_diff_diff = int(n_pos / 2)
        self.n_same_diff = int(n_neg / 2)
        self.n_diff_same = int(n_neg / 2)
        self.flatten_root = flatten_root
        self.flatten_leaves = flatten_leaves
        self.intermediate = intermediate

    def create(self):
        """Main interface

        Attributes
        ----------
        data : list
            Shuffled version of the raw instances, ignoring
            `self.flatten_root` and `self.flatten_leaves`.
            Thus, these are all of the form `(((a, b), (c, d)), label)`

        X : np.array
            The dimensionality depends on `self.flatten_root` and
            `self.flatten_leaves`.

            If both are False, then

            `X.shape == (n_pos+n_neg, 2, 2, embed_dim)`

            If `self.flatten_root`, then

            `X.shape == (n_pos+n_neg, embed_dim*4)`

            If only `self.flatten_leaves`, then

            `X.shape == (n_pos+n_neg, 2, embed_dim*2)`

        y : list
            Containing `POS_LABEL` and `NEG_LABEL`. Length: n_pos+n_neg

        Returns
        -------
        self.X, self.y

        """
        self.data = []
        self.data += self._create_same_same()
        self.data += self._create_diff_diff()
        self.data += self._create_same_diff()
        self.data += self._create_diff_same()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten_root or self.flatten_leaves:
            data = [((np.concatenate(x1), np.concatenate(x2)), label)
                    for (x1, x2), label in data]
        if self.flatten_root:
            data = [(np.concatenate(x), label) for x, label in data]
        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y
        return self.X, self.y

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for root_pair, label in self.data
                               for pair in root_pair for x in pair}
        other_vecs = {tuple(x) for root_pair, label in other_dataset.data
                               for pair in root_pair for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."

    def _create_same_same(self):
        data = []
        for _ in range(self.n_same_same):
            left = self._create_same_pair()
            right = self._create_same_pair()
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_diff_diff(self):
        data = []
        for _ in range(self.n_diff_diff):
            left = self._create_diff_pair()
            right = self._create_diff_pair()
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_same_diff(self):
        data = []
        for _ in range(self.n_same_diff):
            left = self._create_same_pair()
            right = self._create_diff_pair()
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data

    def _create_diff_same(self):
        data = []
        for _ in range(self.n_diff_same):
            left = self._create_diff_pair()
            right = self._create_same_pair()
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data

    def _create_same_pair(self):
        vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        vec1 = randvec(self.embed_dim)
        vec2 = randvec(self.embed_dim)
        assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)


class PremackDatasetLeafFlattened(PremackDataset):
    def __init__(self, embed_dim=50, n_pos=500, n_neg=500):
        super().__init__(
            embed_dim=embed_dim,
            n_pos=n_pos,
            n_neg=n_neg,
            flatten_leaves=True,
            flatten_root=False,
            intermediate=False)
