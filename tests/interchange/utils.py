import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys


STYLE_COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

# Adapted from https://personal.sron.nl/~pault/
ALT_COLORS = ['#000000', '#EE3377', '#CCBB44', '#228833']


def progress_bar(msg):
    sys.stderr.write('\r')
    sys.stderr.write(msg)
    sys.stderr.flush()


def randvec(embed_dim=50, lower=-0.5, upper=0.5):
    return np.array([random.uniform(lower, upper) for i in range(embed_dim)])


def tsne_viz(
        df,
        colors=None,
        output_filename=None,
        figsize=(10, 10),
        random_state=None,
        use_names=True):
    """2d plot of `df` using t-SNE, with the points labeled by `df.index`,
    aligned with `colors` (defaults to all black).

    Adapted from https://github.com/cgpotts/cs224u/

    Parameters
    ----------
    df : pd.DataFrame
        The matrix to visualize.
    colors : list of colornames or None (default: None)
        Optional list of colors for the vocab. The color names just
        need to be interpretable by matplotlib. If they are supplied,
        they need to have the same length as `df.index`. If `colors=None`,
        then all the words are displayed in black.
    output_filename : str (default: None)
        If not None, then the output image is written to this location.
        The filename suffix determines the image type. If `None`, then
        `plt.plot()` is called, with the behavior determined by the
        environment.
    figsize : (int, int) (default: (40, 50))
        Default size of the output in display units.
    random_state : int or None
        Optionally set the `random_seed` passed to `PCA` and `TSNE`.
    use_names : bool
        If True, the index values are printed. If false, dots are printed.
    """
    # Colors:
    vocab = df.index
    if not colors:
        colors = ['black' for i in vocab]
    # Recommended reduction via PCA or similar:
    n_components = 50 if df.shape[1] >= 50 else df.shape[1]
    dimreduce = PCA(n_components=n_components, random_state=random_state)
    X = dimreduce.fit_transform(df)
    # t-SNE:
    tsne = TSNE(n_components=2, random_state=random_state)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(xvals, yvals, marker='', linestyle='')
    if use_names:
        # Text labels:
        for word, x, y, color in zip(vocab, xvals, yvals, colors):
            try:
                ax.annotate(word, (x, y), fontsize=8, color=color)
            except UnicodeDecodeError:  ## Python 2 won't cooperate!
                pass
    else:
        for x, y, color in zip(xvals, yvals, colors):
            ax.plot(x, y, marker='.', color=color)
    plt.xticks([])
    plt.yticks([])
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()


def fix_random_seeds(
        seed=42,
        set_system=True,
        set_torch=True,
        set_torch_cudnn=True):
    """Fix random seeds for reproducibility.

    Adapted from https://github.com/cgpotts/cs224u/

    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    set_torch_cudnn: bool
        Flag for whether to enable cudnn deterministic mode.
        Note that deterministic mode can have a performance impact, depending on your model.
        https://pytorch.org/docs/stable/notes/randomness.html

    Notes
    -----
    The function checks that PyTorch and TensorFlow are installed
    where the user asks to set seeds for them. If they are not
    installed, the seed-setting instruction is ignored. The intention
    is to make it easier to use this function in environments that lack
    one or both of these libraries.

    Even though the random seeds are explicitly set,
    the behavior may still not be deterministic (especially when a
    GPU is enabled), due to:

    * CUDA: There are some PyTorch functions that use CUDA functions
    that can be a source of non-determinism:
    https://pytorch.org/docs/stable/notes/randomness.html

    * PYTHONHASHSEED: On Python 3.3 and greater, hash randomization is
    turned on by default. This seed could be fixed before calling the
    python interpreter (PYTHONHASHSEED=0 python test.py). However, it
    seems impossible to set it inside the python program:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

    """
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
