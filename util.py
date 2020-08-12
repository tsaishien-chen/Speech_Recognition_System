import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH = "./data/"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11, n_mfcc=20):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    #mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc)
    
    # pad on the first and second deltas while we're at it
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc)
    delta1_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc = np.vstack((mfcc,delta1_mfcc,delta2_mfcc))
    
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, mfcc.shape[1]-max_len:]
    
    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels = get_labels(path)[0]

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
 
        np.save('./tmp/' + label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.8):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load('./tmp/' + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load('./tmp/' + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i+1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1-split_ratio), shuffle=True)


