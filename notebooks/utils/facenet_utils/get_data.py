import os
import sys
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
download_folder = os.path.join(root_folder, 'lfw_original')
selection_folder = os.path.join(root_folder, 'lfw_selection')
download_path = os.path.join(download_folder, 'lfw-deepfunneled.tgz')

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

if not os.path.exists(selection_folder):
    os.makedirs(selection_folder)
    
# if not os.path.exists(download_path):
    download_and_uncompress_tarball(database_url, download_folder)

extracted_folder = os.path.join(download_folder, 'lfw-deepfunneled')

# images are organized into separate folders for each person
# get a list of subfolders 
subfolders = [x[0] for x in os.walk(extracted_folder)]

# first item is root the folder itself
subfolders.pop(0) 


def select_persons_with_images(extracted_folder, selection_folder):
    people_list = []

    # Get list of subfolders
    subfolders = [x[0] for x in os.walk(extracted_folder)]
    subfolders.pop(0)  # Remove the root folder

    for path in subfolders:
        image_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        people_list.append((path.split(os.sep)[-1], image_count))
    
    # Sort from max to min images per person
    people_list = sorted(people_list, key=lambda x: x[1], reverse=True)

    for person, image_count in people_list:
        if image_count >= 5:
            file_list = []

            # Create new folder in selected images path
            newpath = os.path.join(selection_folder, person.split(os.sep)[-1])

            if not os.path.exists(newpath):
                os.makedirs(newpath)

            # Copy / paste first 5 images to the new location
            person_full_path = os.path.join(extracted_folder, person)
            files = [os.path.join(person_full_path, f) for f in os.listdir(person_full_path) if os.path.isfile(os.path.join(person_full_path, f))]
            files = files[0:5]  # Select first 5 images
            for file in files:
                filename = os.path.basename(file)
                shutil.copyfile(file, os.path.join(newpath, filename))
                file_list.append(os.path.join(newpath, filename))

select_persons_with_images(extracted_folder, selection_folder)
