import subprocess
import shutil
import numpy as np
from scipy.io import loadmat
import dill
import gzip
import zipfile
from pathlib import Path
import argparse


# Character used for ascii art, sorted in order of increasing sparsity
ascii_art_chars = \
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Maps a relative "sparsity" or "lightness" value in [0, 1) to a character. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(ascii_art_chars)
    bin_id = int(value * n_bins)
    return ascii_art_chars[bin_id]


def image_to_string(array):
    """ Convert an image stored as an array to an ascii art string """
    if array.ndim == 1:
        array = array.reshape(-1, int(np.sqrt(array.shape[0])))
    array = array / array.max()
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'


def maybe_download_emnist(path):
    """
    Download the data if its not already in place.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    path = Path(path).expanduser()
    filename = path / 'emnist'
    if not filename.exists():
        zip_filename = path / 'emnist.zip'
        if not zip_filename.exists():
            print("{} not found, downloading...".format(zip_filename))
            subprocess.run('wget -P {} {}'.format(path, emnist_url).split())
            shutil.move(str(path / 'matlab.zip'), str(path / 'emnist.zip'))

        print("Extracting {}...".format(zip_filename))
        zip_ref = zipfile.ZipFile(str(zip_filename), 'r')
        zip_ref.extractall(str(path))
        zip_ref.close()
        shutil.move(str(path / 'matlab'), str(path / 'emnist'))
    else:
        print("Data found, skipping download.")

    return filename


def process_data(path):
    """
    Download emnist data if it hasn't already been downloaded. Do some
    post-processing to put it in a more useful format. End result is a directory
    called `emnist-byclass` which contains a separate pklz file for each emnist
    class.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    filename = 'emnist-byclass.mat'
    emnist_path = maybe_download_emnist(path)

    print("Processing...")
    mat_path = emnist_path / filename
    dir_name = mat_path.parent / mat_path.stem
    try:
        shutil.rmtree(str(dir_name))
    except FileNotFoundError:
        pass

    dir_name.mkdir(exist_ok=False, parents=False)

    emnist = loadmat(str(mat_path))
    train, test, _ = emnist['dataset'][0, 0]
    train_x, train_y, _ = train[0, 0]
    test_x, test_y, _ = test[0, 0]

    y = np.concatenate((train_y, test_y), 0)
    x = np.concatenate((train_x, test_x), 0)

    # Give images the right orientation so that plt.imshow(x[0]) just works.
    x = np.moveaxis(x.reshape(-1, 28, 28), 1, 2).reshape(-1, 28**2)
    x = x.astype('f') / 255.0

    for i in sorted(set(y.flatten())):
        keep = train_y == i
        x_i = x[keep.flatten(), :]
        if i >= 36:
            char = chr(i-36+ord('a'))
        elif i >= 10:
            char = chr(i-10+ord('A'))
        else:
            char = str(i)

        print(char)
        print(image_to_string(x_i[0, :]))

        path_i = dir_name / (char + '.pklz')
        with gzip.open(str(path_i), 'wb') as f:
            dill.dump(x_i, f, protocol=dill.HIGHEST_PROTOCOL)

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()

    process_data(args.path)
