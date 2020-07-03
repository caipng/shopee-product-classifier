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
        print('wrong number of files, please delete the folder and run get_data.py again')
        return

    if not os.path.isdir(DATA_DIR):
        dataset.process()


if __name__ == '__main__':
    main()
