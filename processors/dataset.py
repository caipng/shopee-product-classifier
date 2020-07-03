import requests
import os
from os.path import join
from math import ceil
import zipfile
import cv2
import numpy as np
from constants import *


def download():
    print("mkdir " + ORIGINAL_DATA_DIR)
    os.makedirs(ORIGINAL_DATA_DIR, exist_ok=True)

    mirrors = ['1K9eByBYt9koE3Fy4vdtS-Rjd8DMDrgW7',
               '1p86o46MtZhhmQeBTlpdqX9eEqoTIwy9p',
               '1mFQUaSH-nwVF9wF60PtN-6Sgi4yPEkZT',
               '1pM5LMNHyGVc4RFtgYaue1-rao8oStKz3',
               '1LT3Rw2ErnIqMNTFoUTH-j1UkAggjWvtT']

    dataset_filepath = join(ORIGINAL_DATA_DIR, "dataset.zip")
    URL = "https://docs.google.com/uc?export=download"
    download_success = False
    for file_id in mirrors:
        print('downloading ' + file_id)
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)

        if 'quota exceeded' in response.text.lower():
            print('file {0} quota exceeded, trying other mirror'.format(file_id))
            continue

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
                break

        with open(dataset_filepath, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

        print('download finished')
        download_success = True
        break

    if not download_success:
        print('download failed!')
        raise RuntimeError

    print('extracting ' + dataset_filepath)
    with zipfile.ZipFile(dataset_filepath, 'r') as z:
        z.extractall(dataset_filepath)

    print('deleting zipfile')
    os.remove(dataset_filepath)


def check_files():
    print('checking number of files')
    expected_num = 105392
    n = 0
    for root, _, files in os.walk(join(ORIGINAL_DATA_DIR, "train", "train")):
        n += len([f for f in files if f.endswith('jpg')])
    print('expected: {0}, actual: {1}'.format(expected_num, n))
    assert n == expected_num, 'wrong number of files'


def process():
    print('processing data')

    # load in predictions of an intermediate model
    loaded = np.load(join('predictions', 'predictions.npz'))
    preds, files, labels = loaded['preds'], loaded['files'], loaded['y']
    # files_table enable lookup of index of a filepath in O(1)
    files_table = {files[i]: i for i in range(files.shape[0])}

    num_processed, num_ignored = 0, 0
    for root, _, files in os.walk(join(ORIGINAL_DATA_DIR, "train", "train")):
        if len(files) == 0:
            continue

        cat = root.split('/')[-1]
        destination_dir = join(DATA_DIR, "train", cat)
        os.makedirs(destination_dir, exist_ok=True)

        for file_path in files:
            if not file_path.endswith("jpg"):
                continue

            num_processed += 1
            if num_processed % 10000 == 0:
                print('processed {0} images'.format(num_processed))

            # we chose to ignore images where P_predicted < 0.1
            # 0.1 is an arbiturary threshold
            fi = files_table[file_path]
            if preds[fi][int(labels[fi])] < 0.1:
                num_ignored += 1
                continue

            im = cv2.imread(join(root, file_path), cv2.IMREAD_COLOR)
            height, width, _ = im.shape

            if height == width:
                im = cv2.resize(im, INPUT_SIZE)
            else:
                if width < height:
                    new_width = INPUT_SIZE[0]
                    new_height = ceil(INPUT_SIZE[1] * (height/width))
                else:
                    new_width = ceil(INPUT_SIZE[0] * (width/height))
                    new_height = INPUT_SIZE[1]

                im = cv2.resize(im, (new_width, new_height))

                if new_width > new_height:
                    x = ceil((new_width-INPUT_SIZE[0])/2)-1
                    im = im[:, x:x+INPUT_SIZE[0], :]
                elif new_height > new_width:
                    y = ceil((new_height-INPUT_SIZE[1])/2)-1
                    im = im[y:y+INPUT_SIZE[1], :, :]

            cv2.imwrite(join(destination_dir, file_path), im)

    print('processed {} images total, ignored {} images'.format(
        num_processed, num_ignored))
