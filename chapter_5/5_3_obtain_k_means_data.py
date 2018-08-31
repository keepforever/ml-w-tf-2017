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
#Example of a spectrogram
y, sr = librosa.load('audio_dataset/cough_1.wav')
ps = librosa.feature.melspectrogram(y=y, sr=sr)
ps.shape
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

my_sound_data = []
my_chroma_data = []
for sound_file_name in filenames:
    y, sr = librosa.load(sound_file_name, duration=1.0)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, )
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    my_sound_data.append(ps)
    my_chroma_data.append(chroma.flatten())
    print(sound_file_name, ps.shape, '\n')

print('chroma', my_chroma_data[0].shape)

single_record = my_chroma_data[0]

# print('notes?', len(my_chroma_data[0][0]))
# print('time', len(my_chroma_data[0][1]))

print('single_record', single_record)

# implement k-means

k = 2

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


clustering = KMeans(n_clusters=k, random_state=5)

clustering.fit(my_chroma_data)
i = 0
for sample in my_chroma_data:
    prediction = clustering.predict(sample)
    print('fileName: ', filenames[i], '\n', 'prediction', prediction)
    i += 1
