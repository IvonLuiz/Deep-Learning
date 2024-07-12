import numpy as np
import os
import sys
import random
import tensorflow as tf
from pathlib import Path
from six.moves import urllib
import tarfile
import shutil


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.
    Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


# URL for sourcing the funneled images
database_url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'

root_folder = '../../../datasets/faces'
download_folder = root_folder + '/'+ 'lfw_original'
selection_folder = root_folder + '/' + 'lfw_selection'
download_path = download_folder + '/lfw-deepfunneled.tgz'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

if not os.path.exists(selection_folder):
    os.makedirs(selection_folder)
    
if not os.path.exists(download_path):
    download_and_uncompress_tarball(database_url, download_folder)

extracted_folder = download_folder + '/lfw-deepfunneled'

# images are organized into separate folders for each person
# get a list of subfolders 
subfolders = [x[0] for x in os.walk(extracted_folder)]

# first item is root the folder itself
subfolders.pop(0) 