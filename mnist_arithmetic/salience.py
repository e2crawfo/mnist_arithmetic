from pathlib import Path
import argparse
import numpy as np

from mnist_arithmetic.utils import image_to_string
from mnist_arithmetic.emnist import PatchesDataset, load_emnist


class SalienceDataset(PatchesDataset):
    """ A dataset for detecting salience.

    Parameters
    ----------
    data_path: str
        Directory containing data. Assumes there is a sub-directory
        called `emnist/emnist-byclass' inside it.
    n_examples: int
        Number of input-output pairs to create.
    min_digits: int (>= 0)
        Minimum number of digits in each image.
    max_digits: int (>= 0)
        Maximum number of digits in each image.
    sub_image_shape: (int, int)
        Shape of component images used to create input images.
    image_shape: (int, int)
        Shape of the input images.
    output_shape: (int, int)
        Shape of the output salience maps.
    max_overlap: int
        Maximum number of pixels that are permitted to be occupied by two separate emnist images.
        Setting to higher values allows more digits to be packed into an image of a fixed size.
    std: float > 0
        Standard deviation of the salience bumps.
    flatten_output: bool
        If True, output ``labels`` are flattened.
    point: bool
        If True, salience represented by single pixel rather than a gaussian.

    """
    def __init__(
            self, data_path, n_examples, classes=None, min_digits=1, max_digits=1,
            sub_image_shape=(14, 14), image_shape=(42, 42), output_shape=(14, 14),
            max_overlap=1, std=0.1, flatten_output=False, point=False):
        if not classes:
            classes = list(range(10))

        data_path = Path(data_path).expanduser()

        self.min_digits = min_digits
        self.max_digits = max_digits

        self.output_shape = output_shape
        self.std = std

        self.X, self.Y, _ = load_emnist(data_path, classes, shape=sub_image_shape)

        super(SalienceDataset, self).__init__(n_examples, image_shape, max_overlap)

        y = []

        for x, pc in zip(self.x, self.patch_centres):
            _y = np.zeros(output_shape)
            for centre in pc:
                if point:
                    pixel_y = int(centre[0] * output_shape[0] / image_shape[0])
                    pixel_x = int(centre[1] * output_shape[1] / image_shape[1])
                    _y[pixel_y, pixel_x] = 1.0
                else:
                    kernel = gaussian_kernel(
                        output_shape, (centre[0]/image_shape[0], centre[1]/image_shape[1]), std)
                    _y = np.maximum(_y, kernel)
            y.append(_y)

        y = np.array(y)
        if flatten_output:
            y = y.reshape(y.shape[0], -1)
        self.y = y

        del self.X
        del self.Y

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        indices = np.random.randint(0, self.Y.shape[0], n)
        images = [self.X[i] for i in indices]
        y = 0
        return images, y

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        m = int(np.ceil(np.sqrt(n)))
        fig, axes = plt.subplots(m, 2 * m)
        for i, s in enumerate(axes[:, :m].flatten()):
            s.imshow(self.x[i, :].reshape(self.image_shape))
        for i, s in enumerate(axes[:, m:].flatten()):
            s.imshow(self.y[i, :].reshape(self.image_shape))


def gaussian_kernel(shape, mu, std):
    """ creates gaussian kernel with side length l and a sigma of sig """
    axy = (np.arange(shape[0]) + 0.5) / shape[1]
    axx = (np.arange(shape[1]) + 0.5) / shape[1]
    yy, xx = np.meshgrid(axx, axy, indexing='ij')

    kernel = np.exp(-((xx - mu[1])**2 + (yy - mu[0])**2) / (2. * std**2))

    return kernel


def test_salience(
        path, sub_image_shape=(14, 14), n_examples=20,
        min_digits=2, max_digits=3, image_shape=(40, 40),
        max_overlap=1):

    dataset = SalienceDataset(
        path, n_examples, [0, 1, 2], min_digits=min_digits,
        max_digits=max_digits, sub_image_shape=sub_image_shape,
        image_shape=image_shape, max_overlap=max_overlap,
        std=0.05, output_shape=(20, 20))

    batch_x, batch_y = dataset.next_batch()

    for x, y in zip(batch_x, batch_y):
        print(image_to_string(y))
        print(image_to_string(x))
    print(batch_x.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    test_salience(args.path)
