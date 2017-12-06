from pathlib import Path
import os
from skimage.transform import resize
import scipy
import numpy as np
import argparse

from mnist_arithmetic.utils import image_to_string


def omniglot_classes(path):
    omniglot_dir = str(Path(path) / 'omniglot')
    alphabets = os.listdir(omniglot_dir)
    classes = []
    for ab in alphabets:
        n_characters = len(os.listdir(os.path.join(omniglot_dir, ab)))
        classes.extend(["{},{}".format(ab, i+1) for i in range(n_characters)])
    return classes


# Class spec: alphabet,character
def load_omniglot(
        path, classes, include_blank=False, shape=None, one_hot=False, indices=None, show=False):
    """ Load omniglot data from disk by class.

    Elements of `classes` pick out which omniglot classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    Returned images are arrays of floats in the range 0-255. White text on black background
    (with 0 corresponding to black). Returned X array has shape (n_images,) + shape.

    Parameters
    ----------
    path: str
        Path to data directory, assumed to contain a sub-directory called `omniglot`.
    classes: list of strings, each giving a class label
        Each character is the name of a class to load.
    balance: boolean
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: boolean
        If True, includes an additional class that consists of blank images.
    shape: (int, int)
        Shape of returned images.
    one_hot: bool
        If True, labels are one-hot vectors instead of integers.
    indices: list of int
        The image indices within the classes to include. For each class there are 20 images.
    show: bool
        If True, prints out an image from each class.

    """
    omniglot_dir = os.path.join(path, 'omniglot')
    classes = list(classes)[:]
    if not indices:
        indices = list(range(20))
    for idx in indices:
        assert 0 <= idx < 20
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        alphabet, character = cls.split(',')
        char_dir = os.path.join(omniglot_dir, alphabet, "character{:02d}".format(int(character)))
        files = os.listdir(char_dir)
        class_id = files[0].split("_")[0]

        for idx in indices:
            f = os.path.join(char_dir, "{}_{:02d}.png".format(class_id, idx + 1))
            _x = scipy.misc.imread(f)
            _x = 255. - _x
            if shape:
                _x = resize(_x, shape, mode='edge')

            x.append(np.float32(_x))
            y.append(i)
        if show:
            print(cls)
            print(image_to_string(x[-1]))
        class_map[cls] = i

    x = np.array(x)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count,) + shape)
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1), dtype=y.dtype)), axis=0)
        blank_symbol = ' '
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes))).astype('f')
        _y[np.arange(y.shape[0]), y.flatten()] = 1.0
        y = _y

    return x, y, class_map


def test(path):
    np.random.seed(10)

    classes = omniglot_classes(path)
    classes = np.random.choice(classes, 20, replace=False)

    x, y, _ = load_omniglot(path, classes, shape=(28, 28), show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    test(args.path)
