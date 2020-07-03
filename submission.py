from const import *
import os
import csv
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
from constants import *

models = []
for model_file in os.listdir(FINAL_MODELS_DIR):
    models.append(load_model(os.path.join(FINAL_MODELS_DIR, model_file)))

xgb_model_path = os.path.join('ensemble', "model.pickle.dat")
xgb_model = pickle.load(open(xgb_model_path, "rb"))

gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=INPUT_SIZE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classified = 0
with open('submission.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['filename', 'category'])

    for test_images in gen:
        i = (gen.batch_index-1)*gen.batch_size
        batch_files = gen.filenames[i:i+gen.batch_size]

        p_all = [[] for _ in range(BATCH_SIZE)]
        for model in models:
            preds = model.predict(test_images)
            for i, p in enumerate(preds):
                p_all[i].extend(p)

        columns = ['m{}_{}'.format(i, j)
                   for i in range(len(models)) for j in range(42)]
        x = pd.DataFrame(p_all, columns=columns)
        final_preds = xgb_model.predict(x)

        for i, f in enumerate(batch_files):
            num_classified += 1
            if num_classified % 1000:
                print('classified {}'.format(num_classified))

            fid = f.split('/')[1]

            if len(fid) != 36:
                print(f)
                continue

            y = int(final_preds[i])
            y = str(y) if y > 9 else '0' + str(y)
            writer.writerow([fid, y])
