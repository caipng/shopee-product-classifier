from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from processors.augmentation import get_generators
from modules.lr import CosineAnnealingLRSchedule
from constants import *

model = load_model(os.path.join('models', 'model2.h5'))

# freeze first 3 convulation blocks
for layer in model.layers[:81]:
    layer.trainable = False
for layer in model.layers[81:]:
    layer.trainable = True

model.compile(
    optimizer=SGD(momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

print(model.summary())

n_epochs = 24
n_cycles = 2
lrate_max = 0.01
checkpoint_filepath = os.path.join('models', 'model3.h5')
generator, _ = get_generators()

model.fit(
    generator,
    steps_per_epoch=38,
    class_weight=CLASS_WEIGHTS,
    epochs=n_epochs,
    verbose=1,
    callbacks=[
        CosineAnnealingLRSchedule(
            n_epochs, n_cycles, lrate_max,
            verbose=1, save_prefix='model3'),
        ModelCheckpoint(
            checkpoint_filepath, monitor='loss',
            verbose=1, save_best_only=True, mode='min')
    ],
)
