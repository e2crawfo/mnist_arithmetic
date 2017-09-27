from pathlib import Path
import dill
import gzip
import argparse

import matplotlib.pyplot as plt
import numpy as np

from download import image_to_string


def load_emnist(
        path, classes, balance=False, include_blank=False, downsample_factor=None, one_hot=False, max_examples=None):
    """ Load emnist data from disk by class.

    Elements of `classes` pick out which emnist classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    Parameters
    ----------
    path: str
        Path to 'emnist-byclass' directory.
    classes: list of character from the set (0-9, A-Z, a-z)
        Each character is the name of a class to load.
    balance: boolean
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: boolean
        If True, includes an additional class that consists of blank images.
    downsample_factor: int
        The emnist digits are stored as 28x28. With downsample_factor=2, the
        images are returned as 14x14, with downsample_factor=4 the images are returned
        as 7x7.
    one_hot: bool
        If True, labels are one-hot vectors instead of integers.
    max_examples: int
        Maximum number of examples returned. If not supplied, return all available data.

    """
    emnist_dir = Path(path) / 'emnist/emnist-byclass'
    classes = list(classes)[:]
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        # print("Loading emnist class {}...".format(cls))

        with gzip.open(str(emnist_dir / (str(cls) + '.pklz')), 'rb') as f:
            _x = dill.load(f)
            n_examples = _x.shape[0]
            if downsample_factor is not None and downsample_factor > 1:
                s = int(np.sqrt(_x.shape[-1]))
                _x = _x.reshape((-1, s, s))
                _x = _x[:, ::downsample_factor, ::downsample_factor]
                _x = _x.reshape((n_examples, -1))
            x.append(_x)
            y.extend([i] * x[-1].shape[0])
        class_map[cls] = i
    x = np.concatenate(x, axis=0)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count, x.shape[1]))
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1))), axis=0)
        blank_symbol = ' '
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

    if balance:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        keep_x, keep_y = [], []
        for i, cls in enumerate(classes):
            keep_indices, _ = np.nonzero(y == class_map[cls])
            keep_indices = keep_indices[:class_count]
            keep_x.append(x[keep_indices, :])
            keep_y.append(y[keep_indices, :])
        x = np.concatenate(keep_x, 0)
        y = np.concatenate(keep_y, 0)

    if max_examples is not None:
        x = x[:max_examples]
        y = y[:max_examples]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes)))
        _y[np.arange(y.shape[0]), y.flatten()] = 1.0
        y = _y

    return x, y, class_map


class Rect(object):
    def __init__(self, x, y, w, h):
        self.left = x
        self.right = x+w
        self.top = y+h
        self.bottom = y

    def intersects(self, r2):
        r1 = self
        h_overlaps = (r1.left <= r2.right) and (r1.right >= r2.left)
        v_overlaps = (r1.bottom <= r2.top) and (r1.top >= r2.bottom)
        return h_overlaps and v_overlaps

    def __str__(self):
        return "<%d:%d %d:%d>" % (self.left, self.right, self.top, self.bottom)


class RegressionDataset(object):

    def __init__(self, x, y, shuffle=True):
        self.x = x
        self.y = y
        assert self.x.shape[0] == self.y.shape[0]
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def n_examples(self):
        return self.x.shape[0]

    @property
    def obs_shape(self):
        return self.x.shape[1:]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completion(self):
        return self.epochs_completed + self.index_in_epoch / self.n_examples

    def next_batch(self, batch_size=None, advance=True):
        """ Return the next ``batch_size`` examples from this data set.

        If ``batch_size`` not specified, return rest of the examples in the current epoch.

        """
        start = self._index_in_epoch

        if batch_size is None:
            batch_size = self.n_examples - start
        elif batch_size > self.n_examples:
            raise Exception(
                "Too few examples ({}) to satisfy batch size "
                "of {}.".format(self.n_examples, batch_size))

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(self.n_examples)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        if start + batch_size >= self.n_examples:
            # Finished epoch

            # Get the remaining examples in this epoch
            x_rest_part = self._x[start:]
            y_rest_part = self._y[start:]

            # Shuffle the data
            if self.shuffle and advance:
                perm = np.arange(self.n_examples)
                np.random.shuffle(perm)
                self._x = self.x[perm]
                self._y = self.y[perm]

            # Start next epoch
            end = batch_size - len(x_rest_part)
            x_new_part = self._x[:end]
            y_new_part = self._y[:end]
            x = np.concatenate((x_rest_part, x_new_part), axis=0)
            y = np.concatenate((y_rest_part, y_new_part), axis=0)

            if advance:
                self._index_in_epoch = end
                self._epochs_completed += 1
        else:
            # Middle of epoch
            end = start + batch_size
            x, y = self._x[start:end], self._y[start:end]

            if advance:
                self._index_in_epoch = end

        return x, y


class PatchesDataset(RegressionDataset):
    def __init__(self, n_examples, image_width, max_overlap, **kwargs):
        self.image_width = image_width
        self.max_overlap = max_overlap

        x, y = self._make_dataset(n_examples)
        super(PatchesDataset, self).__init__(x, y, **kwargs)

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _make_dataset(self, n_examples):
        max_overlap, image_width = self.max_overlap, self.image_width
        if n_examples == 0:
            return np.zeros((0, image_width, image_width)).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []

        for j in range(n_examples):
            images, y = self._sample_patches()
            image_shapes = [img.shape for img in images]

            # Sample rectangles
            n_rects = len(images)
            i = 0
            while True:
                rects = [
                    Rect(
                        np.random.randint(0, image_width-m+1),
                        np.random.randint(0, image_width-n+1), m, n)
                    for m, n in image_shapes]
                area = np.zeros((image_width, image_width), 'f')

                for rect in rects:
                    area[rect.left:rect.right, rect.bottom:rect.top] += 1

                if (area >= 2).sum() < max_overlap:
                    break

                i += 1

                if i > 1000:
                    raise Exception(
                        "Could not fit rectangles. "
                        "(n_rects: {}, image_width: {}, max_overlap: {})".format(
                            n_rects, image_width, max_overlap))

            # Populate rectangles
            o = np.zeros((image_width, image_width), 'f')
            for image, rect in zip(images, rects):
                o[rect.left:rect.right, rect.bottom:rect.top] += image

            new_X.append(np.uint8(255*np.minimum(o, 1)))
            new_Y.append(y)

        new_X = np.array(new_X).astype('f')
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y

    def visualize(self, n=9):
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)
        size = int(np.sqrt(self.x.shape[1]))
        for i, s in enumerate(subplots.flatten()):
            s.imshow(self.x[i, :].reshape(size, size))
            s.set_title(str(self.y[i, 0]))


class MnistArithmeticDataset(PatchesDataset):
    """ A dataset for the MNIST arithmetic task.

    Parameters
    ----------
    data_path: str
        Directory containing data. Assumes there is a sub-directory
        called `emnist/emnist-byclass' inside it.
    n_examples: int
        Number of input-output pairs to create.
    reductions: dict (str -> function) or single function
        If a dict, then it should map from characters to reduction functions.
        For each image, one of these operations will be randomly sampled.
        Otherwise, should be a single reduction function, and no operation symbol
        will be added to the images.
    min_digits: int (> 0)
        Minimum number of digits in each image.
    max_digits: int (> 0)
        Maximum number of digits in each image.
    base: int (<= 10, > 0)
        Base of digits that appear in image (e.g. if `base == 2`, only digits 0 and 1 appear).
    downsample_factor: int
        Factor to shrink the component emnist images by.
    image_width: int
        Width in pixels of the images.
    max_overlap: int
        Maximum number of pixels that are permitted to be occupied by two separate emnist images.
        Setting to higher values allows more digits to be packed into an image of a fixed size.

    """
    def __init__(
            self, data_path, n_examples, reductions, min_digits=1,
            max_digits=1, base=10, downsample_factor=1, image_width=100, max_overlap=200):

        data_path = Path(data_path).expanduser()

        self.min_digits = min_digits
        self.max_digits = max_digits

        self.X, self.Y, _ = load_emnist(
            data_path, [str(i) for i in range(base)],
            downsample_factor=downsample_factor)

        s = int(np.sqrt(self.X.shape[1]))
        self.X = self.X.reshape(-1, s, s)

        if isinstance(reductions, dict):
            self.sample_op = True
            self.eX, self.eY, op_class_map = load_emnist(
                data_path, list(reductions.keys()),
                downsample_factor=downsample_factor)

            s = int(np.sqrt(self.eX.shape[1]))
            self.eX = self.eX.reshape(-1, s, s)
            self._remapped_reductions = {op_class_map[k]: v for k, v in reductions.items()}
        else:
            assert callable(reductions)
            self.func = reductions
            self.sample_op = False
            self.eX = self.eY = None

        super(MnistArithmeticDataset, self).__init__(n_examples, image_width, max_overlap)

        del self.X
        del self.Y
        del self.eX
        del self.eY

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        digit_indices = np.random.randint(0, self.Y.shape[0], n)
        images = [self.X[i] for i in digit_indices]

        if self.sample_op:
            symbol_idx = np.random.randint(0, self.eY.shape[0])
            images.insert(0, self.eX[symbol_idx])
            symbol_class = self.eY[symbol_idx, 0]
            func = self._remapped_reductions[symbol_class]
        else:
            func = self.func

        y = func([self.Y[i] for i in digit_indices])

        return images, y


def test(reductions):
    n_examples = 20
    min_digits = 1
    max_digits = 3
    base = 10
    image_width = 100
    max_overlap = 200

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()

    dataset = MnistArithmeticDataset(
        args.path, n_examples, reductions, min_digits=min_digits, max_digits=max_digits,
        base=base, image_width=image_width, max_overlap=max_overlap)

    batch_x, batch_y = dataset.next_batch()

    for x, y in zip(batch_x, batch_y):
        print("\nCorrect answer is: {}".format(y[0]))
        print(image_to_string(x))


if __name__ == "__main__":
    reductions = {
        'A': lambda x: sum(x),
        'M': lambda x: np.product(x),
        'C': lambda x: len(x),
        'X': lambda x: max(x),
        'N': lambda x: min(x)
    }
    test(reductions)

    test(sum)
