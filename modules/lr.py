import os
from math import floor, cos, pi
from tensorflow.keras.callbacks import Callback
from keras import backend


class CosineAnnealingLRSchedule(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0, save_prefix="", metric='acc'):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
        self.save_prefix = save_prefix
        self.metric = metric

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(
            epoch, self.epochs, self.cycles, self.lr_max)
        print('\nlr set to ' + str(lr))
        backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            filename = "snapshot_model_%d(%s).h5" % (int(
                (epoch + 1) / epochs_per_cycle), str(logs[self.metric]))
            if self.save_prefix != "":
                filename = "{}_{}".format(self.save_prefix, filename)
            os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
            self.model.save(os.path.join(os.getcwd(), 'models', filename))
            print('\n>saved snapshot %s, epoch %d\n' % (filename, epoch))
