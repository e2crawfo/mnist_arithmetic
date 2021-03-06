from pathlib import Path
import dill
import gzip
import argparse
from skimage.transform import resize
import numpy as np
import os
import shutil
import warnings

from mnist_arithmetic.utils import image_to_string, cd


emnist_filenames = [
    '0.pklz',
    '1.pklz',
    '2.pklz',
    '3.pklz',
    '4.pklz',
    '5.pklz',
    '6.pklz',
    '7.pklz',
    '8.pklz',
    '9.pklz',
    'A.pklz',
    'B.pklz',
    'C.pklz',
    'D.pklz',
    'E.pklz',
    'F.pklz',
    'G.pklz',
    'H.pklz',
    'I.pklz',
    'J.pklz',
    'K.pklz',
    'L.pklz',
    'M.pklz',
    'N.pklz',
    'O.pklz',
    'P.pklz',
    'Q.pklz',
    'R.pklz',
    'S.pklz',
    'T.pklz',
    'U.pklz',
    'V.pklz',
    'W.pklz',
    'X.pklz',
    'Y.pklz',
    'Z.pklz',
    'a.pklz',
    'b.pklz',
    'c.pklz',
    'd.pklz',
    'e.pklz',
    'f.pklz',
    'g.pklz',
    'h.pklz',
    'i.pklz',
    'j.pklz',
    'k.pklz',
    'l.pklz',
    'm.pklz',
    'n.pklz',
    'o.pklz',
    'p.pklz',
    'q.pklz',
    'r.pklz',
    's.pklz',
    't.pklz',
    'u.pklz',
    'v.pklz',
    'w.pklz',
    'x.pklz',
    'y.pklz',
    'z.pklz'
]


def _validate_emnist(path):
    path = str(path)
    if not os.path.isdir(path):
        return False

    with cd(path):
        return set(os.listdir(path)) == set(emnist_filenames)


def convert_emnist_and_store(path, new_image_shape):
    if new_image_shape == (28, 28):
        raise Exception("Original shape of EMNIST is (28, 28).")

    print("Converting (28, 28) EMNIST dataset to {}...".format(new_image_shape))

    emnist_dir = Path(path) / 'emnist'
    new_dir = Path(path) / 'emnist_{}_by_{}'.format(*new_image_shape)
    try:
        shutil.rmtree(str(new_dir))
    except FileNotFoundError:
        pass
    new_dir.mkdir(exist_ok=False, parents=False)

    classes = ''.join(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )

    for i, cls in enumerate(sorted(classes)):
        with gzip.open(str(emnist_dir / (str(cls) + '.pklz')), 'rb') as f:
            _x = dill.load(f)

            new_x = []
            for img in _x:
                img = resize(img, new_image_shape, mode='edge')
                new_x.append(img)

            print(cls)
            print(image_to_string(_x[0]))
            _x = np.array(new_x)
            print(image_to_string(_x[0]))

            path_i = new_dir / (cls + '.pklz')
            with gzip.open(str(path_i), 'wb') as f:
                dill.dump(_x, f, protocol=dill.HIGHEST_PROTOCOL)


def load_emnist(
        path, classes, balance=False, include_blank=False,
        shape=None, one_hot=False, n_examples=None, show=False):
    """ Load emnist data from disk by class.

    Elements of `classes` pick out which emnist classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    Pixel values of returned images are integers in the range 0-255, but stored as float32.
    Returned X array has shape (n_images,) + shape.

    Parameters
    ----------
    path: str
        Path to data directory, assumed to contain a sub-directory called `emnist`.
    classes: list of character from the set (0-9, A-Z, a-z)
        Each character is the name of a class to load.
    balance: boolean
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: boolean
        If True, includes an additional class that consists of blank images.
    shape: (int, int)
        Shape of the images.
    one_hot: bool
        If True, labels are one-hot vectors instead of integers.
    n_examples: int
        Maximum number of examples returned. If not supplied, return all available data.
    show: bool
        If True, prints out an image from each class.

    """
    emnist_dir = Path(path) / 'emnist'

    needs_reshape = False
    if shape and shape != (28, 28):
        resized_dir = Path(path) / 'emnist_{}_by_{}'.format(*shape)

        if _validate_emnist(resized_dir):
            emnist_dir = resized_dir
        else:
            needs_reshape = True

    classes = list(classes)[:]
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        with gzip.open(str(emnist_dir / (str(cls) + '.pklz')), 'rb') as f:
            _x = dill.load(f)
            x.append(np.float32(np.uint8(255*np.minimum(_x, 1))))
            y.extend([i] * x[-1].shape[0])
        if show:
            print(cls)
            print(image_to_string(x[-1]))
        class_map[cls] = i
    x = np.concatenate(x, axis=0)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count,) + x.shape[1:])
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1), dtype=y.dtype)), axis=0)
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

    if n_examples is not None:
        x = x[:n_examples]
        y = y[:n_examples]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes))).astype('f')
        _y[np.arange(y.shape[0]), y.flatten()] = 1.0
        y = _y

    if needs_reshape:
        if x.shape[0] > 10000:
            warnings.warn(
                "Performing an online resize of a large number of images ({}), "
                "consider creating and storing the resized dataset.".format(x.shape[0])
            )

        x = [resize(img, shape, mode='edge') for img in np.uint8(x)]
        x = np.float32(np.uint8(255*np.minimum(x, 1)))

    return x, y, class_map


class Rect(object):
    def __init__(self, y, x, h, w):
        self.top = y
        self.bottom = y+h
        self.left = x
        self.right = x+w

    def intersects(self, r2):
        r1 = self
        h_overlaps = (r1.left <= r2.right) and (r1.right >= r2.left)
        v_overlaps = (r1.top <= r2.bottom) and (r1.bottom >= r2.top)
        return h_overlaps and v_overlaps

    def centre(self):
        return (
            self.top + (self.bottom - self.top) / 2.,
            self.left + (self.right - self.left) / 2.
        )

    def __str__(self):
        return "<%d:%d %d:%d>" % (self.top, self.bottom, self.left, self.right)


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
    def __init__(self, n_examples, image_shape, max_overlap, draw_shape=None, draw_offset=None, **kwargs):
        self.image_shape = image_shape
        self.draw_shape = draw_shape or image_shape
        self.draw_offset = draw_offset or (0, 0)

        assert self.draw_offset[0] >= 0
        assert self.draw_offset[1] >= 0

        assert self.draw_offset[0] + self.draw_shape[0] <= self.image_shape[0]
        assert self.draw_offset[1] + self.draw_shape[1] <= self.image_shape[1]

        self.max_overlap = max_overlap

        x, y, self.patch_centres = self._make_dataset(n_examples)
        super(PatchesDataset, self).__init__(x, y, **kwargs)

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _make_dataset(self, n_examples):
        if n_examples == 0:
            return np.zeros((0,) + self.image_shape).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []
        patch_centres = []

        for j in range(n_examples):
            sub_images, y = self._sample_patches()
            sub_image_shapes = [img.shape for img in sub_images]

            # Sample rectangles
            n_rects = len(sub_images)
            i = 0
            while True:
                rects = [
                    Rect(
                        np.random.randint(0, self.draw_shape[0]-m+1),
                        np.random.randint(0, self.draw_shape[1]-n+1), m, n)
                    for m, n in sub_image_shapes]
                area = np.zeros(self.draw_shape, 'f')

                for rect in rects:
                    area[rect.top:rect.bottom, rect.left:rect.right] += 1

                if (area >= 2).sum() < self.max_overlap:
                    break

                i += 1

                if i > 1000:
                    raise Exception(
                        "Could not fit rectangles. "
                        "(n_rects: {}, draw_shape: {}, max_overlap: {})".format(
                            n_rects, self.draw_shape, self.max_overlap))

            patch_centres.append([r.centre() for r in rects])

            # Populate rectangles
            x = np.zeros(self.draw_shape, 'f')
            for image, rect in zip(sub_images, rects):
                patch = x[rect.top:rect.bottom, rect.left:rect.right]
                x[rect.top:rect.bottom, rect.left:rect.right] = np.maximum(image, patch)

            if self.draw_shape != self.image_shape or self.draw_offset != (0, 0):
                _x = np.zeros(self.image_shape, 'f')
                y_start, x_start = self.draw_offset
                y_end, x_end = y_start + self.draw_shape[0], x_start + self.draw_shape[1]
                _x[y_start:y_end, x_start:x_end] = x
                x = _x

            new_X.append(x)
            new_Y.append(y)

        new_X = np.array(new_X).astype('f')
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y, patch_centres

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
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
        called `emnist` inside it.
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
    one_hot: bool
        Whether to specify labels as one-hot vectors.
    largest_digit: int > 0
        Largest digit that gets its own class when doing one-hot encoding. All larger digits
        mapped to a final class. So there are `largest_digit + 2` classes in total.
    base: int (<= 10, > 0)
        Base of digits that appear in image (e.g. if `base == 2`, only digits 0 and 1 appear).
    sub_image_shape: int
        Shape of component EMNIST images.
    draw_shape: int
        Shape of sub-region of the overall image in which digits will be drawn.
    draw_offset: int
        Top left corner of sub-region of the overall image in which digits will be drawn.
    image_shape: int
        Shape of input images.
    max_overlap: int
        Maximum number of pixels that are permitted to be occupied by two separate emnist images.
        Setting to higher values allows more digits to be packed into an image of a fixed size.

    """
    reductions_dict = {
        "sum": sum,
        "prod": np.product,
        "max": max,
        "min": min,
        "len": len,
    }

    def __init__(
            self, data_path, n_examples, reductions, min_digits=1, max_digits=1, one_hot=False, largest_digit=99,
            base=10, sub_image_shape=None, draw_shape=None, draw_offset=None, image_shape=(100, 100), max_overlap=200):

        data_path = Path(data_path).expanduser()

        self.min_digits = min_digits
        self.max_digits = max_digits
        self.one_hot = one_hot
        self.largest_digit = largest_digit

        self.X, self.Y, _ = load_emnist(
            data_path, [str(i) for i in range(base)], shape=sub_image_shape)

        if isinstance(reductions, str):
            if ":" not in reductions:
                reductions = self.reductions_dict[reductions.strip()]
            else:
                _reductions = {}
                delim = ',' if ',' in reductions else ' '
                for pair in reductions.split(delim):
                    char, key = pair.split(':')
                    _reductions[char] = self.reductions_dict[key]
                reductions = _reductions

        if isinstance(reductions, dict):
            self.sample_op = True
            self.eX, self.eY, op_class_map = load_emnist(
                data_path, list(reductions.keys()), shape=sub_image_shape)

            self._remapped_reductions = {op_class_map[k]: v for k, v in reductions.items()}
        else:
            assert callable(reductions)
            self.func = reductions
            self.sample_op = False
            self.eX = self.eY = None

        super(MnistArithmeticDataset, self).__init__(n_examples, image_shape, max_overlap, draw_shape, draw_offset)

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

        if self.one_hot:
            _y = np.zeros(self.largest_digit + 2)
            if y > self.largest_digit:
                _y[-1] = 1.0
            else:
                _y[int(y)] = 1.0
            y = _y
        else:
            y = np.minimum(y, self.largest_digit+1)

        return images, y


def test(*args, **kwargs):
    dataset = MnistArithmeticDataset(*args, **kwargs)

    batch_x, batch_y = dataset.next_batch()

    for x, y in zip(batch_x, batch_y):
        print("\nCorrect answer is: {}".format(y[0]))
        print(image_to_string(x))


if __name__ == "__main__":
    reductions = {
        'A': sum,
        'M': np.product,
        'C': len,
        'X': max,
        'N': min,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    kwargs = dict(
        data_path=args.path, n_examples=10, reductions=reductions, sub_image_shape=(14, 14), image_shape=(100, 100),
        max_overlap=10, min_digits=1, max_digits=2, draw_shape=(40, 40), draw_offset=(25, 25))

    test(**kwargs)

    kwargs.update(reductions=sum)

    test(**kwargs)

    kwargs.update(draw_shape=None, draw_offset=None, reductions=reductions)

    test(**kwargs)
