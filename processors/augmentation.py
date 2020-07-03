import os
from random import randint, uniform
from PIL import Image, ImageEnhance, ImageFilter
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
from tensorflow.keras.applications import resnet50, xception, inception_resnet_v2, vgg16
import numpy as np
from constants import *


def random_crop(img):
    height, width = img.shape[0], img.shape[1]
    dy, dx = INPUT_SIZE
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def custom_augment(img, brightness_range=[0.7, 1.2], blur_range=[1, 2]):
    # contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98)) * 255.

    # convert to PIL Image
    img = Image.fromarray(img.astype(np.uint8))

    # randomly apply brightness change
    if randint(1, 5) == 1:
        img = ImageEnhance.Brightness(img).enhance(
            uniform(brightness_range[0], brightness_range[1]))

    # randomly apply blur
    if randint(1, 5) == 1:
        img = img.filter(ImageFilter.GaussianBlur(
            uniform(blur_range[0], blur_range[1])))

    # convert back to np array
    return np.array(img).astype(np.float32)


def custom_train_generator(generator):
    while True:
        batch_x, batch_y = next(generator)
        batch_augmented = np.zeros(
            (batch_x.shape[0], INPUT_SIZE[0], INPUT_SIZE[1], 3))
        for i in range(batch_x.shape[0]):
            x = batch_x[i]
            # x = random_crop(x)
            x = custom_augment(x)
            batch_augmented[i] = x
        yield (batch_augmented, batch_y)


def get_generators(model='resnet50'):
    supported_models = ['resnet50', 'xception', 'inception_resnet_v2', 'vgg16']
    assert model in supported_models, 'invalid model'

    preprocess_functions = [resnet50.preprocess_input, xception.preprocess_input,
                            inception_resnet_v2.preprocess_input, vgg16.preprocess_input]
    preprocess_input = preprocess_functions[supported_models.index(model)]

    train_generator = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        # channel_shift_range=10,
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=INPUT_SIZE,
        class_mode='sparse',
        shuffle=True,
        # seed=42,
        batch_size=BATCH_SIZE)

    # no more validation data for final stage
    return (train_generator, None)

    test_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=INPUT_SIZE,
        class_mode='sparse',
        shuffle=True,
        # seed=42,
        batch_size=BATCH_SIZE)

    # train_generator = custom_train_generator(train_generator)
    return (train_generator, test_generator)
