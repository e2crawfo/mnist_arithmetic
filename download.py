import shutil
import numpy as np
from scipy.io import loadmat
import dill
import gzip
from pathlib import Path
import argparse
import os
import subprocess
import zipfile
from contextlib import suppress

from mnist_arithmetic.utils import image_to_string


emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'


def maybe_download_emnist(path):
    """
    Download the data if it is not already in place.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    path = Path(path).expanduser()
    emnist_path = path / 'emnist'
    if not (emnist_path / 'emnist-byclass').exists():
        zip_filename = path / 'emnist.zip'
        if not zip_filename.exists():
            print("{} not found, downloading...".format(zip_filename))
            subprocess.run('wget -P {} {}'.format(path, emnist_url).split())
            shutil.move(str(path / 'matlab.zip'), str(path / 'emnist.zip'))

        shutil.rmtree(str(path / 'matlab'), ignore_errors=True)
        shutil.rmtree(str(emnist_path), ignore_errors=True)

        print("Extracting {}...".format(zip_filename))
        zip_ref = zipfile.ZipFile(str(zip_filename), 'r')
        zip_ref.extractall(str(path))
        zip_ref.close()
        shutil.move(str(path / 'matlab'), str(emnist_path))

        os.remove(str(emnist_path / 'emnist-balanced.mat'))
        os.remove(str(emnist_path / 'emnist-bymerge.mat'))
        os.remove(str(emnist_path / 'emnist-digits.mat'))
        os.remove(str(emnist_path / 'emnist-letters.mat'))
        os.remove(str(emnist_path / 'emnist-mnist.mat'))
    else:
        print("Data found, skipping download.")


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
    maybe_download_emnist(path)

    path = Path(path)

    print("Processing...")
    emnist_path = path / 'emnist'
    mat_path = emnist_path / 'emnist-byclass.mat'
    dir_name = emnist_path / 'emnist-byclass'
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
        keep = y == i
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

    try:
        os.remove(str(path / 'emnist.zip'))
    except:
        pass
    os.remove(str(path / 'emnist/emnist-byclass.mat'))

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()

    process_data(args.path)
