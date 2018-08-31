import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
    Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from os import listdir
from os.path import isfile, join

# A Store filenames that match a pattern #
filenames = ["audio_dataset/" + f for f in listdir("audio_dataset") if isfile(
    join("audio_dataset", f)) and f.endswith('.wav')]
count_num_files = tf.size(filenames)

print(filenames)
# Example of a spectrogram
# y, sr = librosa.load('audio_dataset/cough_1.wav')
# ps = librosa.feature.melspectrogram(y=y, sr=sr)
# ps.shape
# librosa.display.specshow(ps, y_axis='mel', x_axis='time')
# plt.show()

for sound_file_name in filenames: 
    y, sr = librosa.load(sound_file_name)
    # print('sr', sr)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    # print(sound_file_name, ps.shape, '\n')

y_harmonic, y_percussive = librosa.effects.hpss(y,)


# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, )

c_data_frame = pd.DataFrame(data=C)
print('\n', '\n', '\n', 'c_data_frame', '\n', c_data_frame)

print('C[0]',  '\n', C[0], '\n', 'len(C[0])',
      len(C[0]), '\n', 'C.shape', '\n', C.shape)
print('C[1][0]', C[1][0],  '\n',)
print('C[2]', C[2], '\n',)
print('C[:0]', C[:0],  '\n',)
# Make a new figure
plt.figure(figsize=(12, 4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax
librosa.display.specshow(C, sr=sr, x_axis='time',
                         y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')
plt.colorbar()

plt.tight_layout()

plt.show()
