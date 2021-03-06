import os
from os.path import join

# training parameters
BATCH_SIZE = 128
INPUT_SIZE = (256, 256)
EPOCHS = 10

# working directories
DATA_DIR = join(os.getcwd(), 'data', 'processed')
ORIGINAL_DATA_DIR = join(os.getcwd(), 'data', 'original')
MODELS_DIR = join(os.getcwd(), 'models')
FINAL_MODELS_DIR = join(os.getcwd(), 'models', 'final')

# class weights are calculated by relative proportions of
# training data avaliable for each category - see `processors/class_weights.ipynb`
CLASS_WEIGHTS = {0: 0.9353247075945548,
                 1: 0.9284040660289273,
                 2: 0.9339323373562302,
                 3: 0.9284040660289273,
                 4: 0.9284040660289273,
                 5: 0.9501992391049566,
                 6: 0.9501992391049566,
                 7: 0.943057568762191,
                 8: 0.9294356261022928,
                 9: 0.93012460729288,
                 10: 0.9388238647497906,
                 11: 1.3616257137689585,
                 12: 0.9325441064571499,
                 13: 0.9356734490962679,
                 14: 0.9346280039017469,
                 15: 0.9534484006368504,
                 16: 0.9416420977396587,
                 17: 1.615889369269923,
                 18: 1.1927168205685317,
                 19: 0.9367212357133969,
                 20: 0.9455449097498834,
                 21: 0.9659261703141611,
                 22: 0.9567198591216891,
                 23: 0.9879827521559805,
                 24: 0.9277176304902738,
                 25: 0.9321976933418241,
                 26: 0.9349762259598325,
                 27: 0.9287476648690565,
                 28: 0.9798813707443149,
                 29: 1.173749387500557,
                 30: 0.9277176304902738,
                 31: 0.9374210648024619,
                 32: 1.1634103803783915,
                 33: 4.379539599434888,
                 34: 0.9655545173051906,
                 35: 0.9441219678239994,
                 36: 0.9342800411303762,
                 37: 1.4547688060731538,
                 38: 0.9388238647497906,
                 39: 0.9370710195952915,
                 40: 0.9360224507557592,
                 41: 0.9427033022074345}
