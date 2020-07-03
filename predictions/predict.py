import os
from os.path import join
from math import ceil
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from constants import *


def predict_by_cat():
    model = load_model('snapshot_model_3_vacc_0.7276785969734192.h5')

    for root, _, files in os.walk(DATA_DIR):
        if len(files) == 0:
            continue

        cat = ''.join(root.split('/')[-1])
        print('found {} files for cat {} at {}'.format(len(files), cat, root))

        preds = []
        for i in range(ceil(len(files)/BATCH_SIZE)):
            batch = files[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            images = []
            for filename in batch:
                im = cv2.imread(join(root, filename))
                images.append(preprocess_input(im))
            preds.extend(list(model.predict(np.array(images))))

        np.savez_compressed(
            'cat{}_preds.npz'.format(cat),
            preds=np.array(preds),
            files=np.array(files))
        print('saved {} predictions to {}'.format(
            len(preds), 'cat{}_preds.npz'.format(cat)))



