import os
from distutils.dir_util import copy_tree
from constants import *
from processors import dataset


def main():
    if not os.path.isdir(ORIGINAL_DATA_DIR):
        dataset.download()

    try:
        dataset.check_files()
    except AssertionError:
        print('wrong number of files, please delete the folder and run main.py again')
        return

    if not os.path.isdir(DATA_DIR):
        dataset.process()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FINAL_MODELS_DIR, exist_ok=True)


if __name__ == '__main__':
    main()
