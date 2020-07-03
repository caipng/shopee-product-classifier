import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from processors.augmentation import get_generators
from modules.lr import CosineAnnealingLRSchedule
from constants import *

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # , activity_regularizer=l1_l2(0.001)
x = Dense(42, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

print(model.summary())

model.compile(
    optimizer=SGD(momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

n_epochs = 75
n_cycles = 5
lrate_max = 0.01
checkpoint_filepath = os.path.join('models', 'model1.h5')
generator, _ = get_generators()

model.fit(
    generator,
    steps_per_epoch=50,
    class_weight=CLASS_WEIGHTS,
    epochs=n_epochs,
    verbose=1,
    callbacks=[
        CosineAnnealingLRSchedule(
            n_epochs, n_cycles, lrate_max,
            verbose=1, save_prefix='model1'),
        ModelCheckpoint(
            checkpoint_filepath, monitor='loss',
            verbose=1, save_best_only=True, mode='min')
    ],
)
