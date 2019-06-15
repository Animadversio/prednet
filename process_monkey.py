'''
Code for processing custom image sequence fetched from video in youtube
'''

import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from kitti_settings import *
import glob
import re

DATA_DIR = "monkey_data"
desired_im_sz = (128, 160)  # image size for utube video seems to be 1280 *720
# categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Download raw zip files by scraping KITTI website
# def download_data():
#     base_dir = os.path.join(DATA_DIR, 'raw/')
#     if not os.path.exists(base_dir): os.mkdir(base_dir)
#     for c in categories:
#         url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
#         r = requests.get(url)
#         soup = BeautifulSoup(r.content)
#         drive_list = soup.find_all("h3")
#         drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
#         print( "Downloading set: " + c)
#         c_dir = base_dir + c + '/'
#         if not os.path.exists(c_dir): os.mkdir(c_dir)
#         for i, d in enumerate(drive_list):
#             print( str(i+1) + '/' + str(len(drive_list)) + ": " + d)
#             url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + d + "/" + d + "_sync.zip"
#             urllib.request.urlretrieve(url, filename=c_dir + d + "_sync.zip")
#
#
# # unzip images
# def extract_data():
#     for c in categories:
#         c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
#         zip_files = list(os.walk(c_dir, topdown=False))[-1][-1]#.next()
#         for f in zip_files:
#             print( 'unpacking: ' + f)
#             spec_folder = f[:10] + '/' + f[:-4] + '/image_03/data*'
#             command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4]
#             os.system(command)

excluded_list = []
# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    # for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
    c_dir = os.path.join(DATA_DIR, 'RAW') # no \
    seq_clip_list ={}
    folders = os.listdir(c_dir) # list(os.walk(c_dir, topdown=False))[-1][-2]
    for folder in folders:
        if folder in excluded_list:
            continue
        filenames = sorted(glob.glob1(os.path.join(c_dir, folder), '*.jpg'))
        num_pat = re.compile("([0-9]+)\.")
        img_ids = [int(num_pat.search(filename).group(1)) for filename in filenames]
        start_id = min(img_ids)
        cur_id = start_id
        start_i = 0
        groups = []
        for i, img_id in enumerate(img_ids):
            if img_id == cur_id:
                cur_id += 1
                if img_id == img_ids[-1]:
                    groups.append(((start_i, i + 1), (start_id, cur_id - 1)))
            else:
                groups.append( ((start_i, i + 1), (start_id, cur_id - 1)) )
                # (start_i, end_i + 1), (start_id, end_id)
                # filename[start_i:i+1] = ['start_id', ... 'end_id']
                start_id = img_id
                start_i = i + 1
                cur_id = img_id + 1  # predictive coding!
        seq_clip_list[folder] = groups
        # splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]
    # TODO!
    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'RAW/', folder)
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    # download_data()
    # extract_data()
    process_data()
