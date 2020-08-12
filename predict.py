from util import *
from keras.models import load_model

import sys

if (len(sys.argv) != 2):
    print('#usage:   python3 predict.py <wavefile>')
    print('#example: python3 predict.py example_wav/yes.wav')
    exit()
    
model = load_model('model/model.h5')

mfcc_max_len = 32
n_mfcc       = 13
mfcc_shape   = (n_mfcc*3, mfcc_max_len)
channel      = 1

sample = wav2mfcc(sys.argv[1], max_len=mfcc_max_len, n_mfcc=n_mfcc)
sample_reshaped = sample.reshape(1, mfcc_shape[0], mfcc_shape[1], channel)
predLabel = get_labels()[0][np.argmax(model.predict(sample_reshaped))]
print('=== predict result ===')
print(predLabel)
print('======================')
