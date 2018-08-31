# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

import pickle
import numpy as np
from autoencoder import Autoencoder
#
# def grayscale(x):
#     gray = np.zeros(len(x)/3)
#     for i in range(len(x)/3):
#         gray[i] = (x[i] + x[2*i] + x[3*i]) / 3


def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


names = unpickle('./cifar-10-batches-py/batches.meta')

print(names.keys())
print(names[b'label_names'])

data, labels = [], []
for i in range(1, 6):
    filename = './cifar-10-batches-py/data_batch_' + str(i)
    batch_data = unpickle(filename)
    if len(data) > 0:
        data = np.vstack((data, batch_data[b'data']))
        labels = np.vstack((labels, batch_data[b'labels']))
    else:
        data = batch_data[b'data']
        labels = batch_data[b'labels']

data = grayscale(data)

x = np.matrix(data)
y = np.array(labels)

horse_indices = np.where(y == 7)[0]

horse_x = x[horse_indices]

print(np.shape(horse_x))  # (5000, 3072)

input_dim = np.shape(horse_x)[1]
hidden_dim = 100
ae = Autoencoder(input_dim, hidden_dim)
ae.train(horse_x)

test_data = unpickle('./cifar-10-batches-py/test_batch')
test_x = grayscale(test_data[b'data'])
test_labels = np.array(test_data[b'labels'])
encoding = ae.classify(test_x, test_labels)
encoding = np.matrix(encoding)
from matplotlib import pyplot as plt

# encoding = np.matrix(np.random.choice([0, 1], size=(hidden_dim,)))

original_img = np.reshape(test_x[7, :], (32, 32))
plt.imshow(original_img, cmap='Greys_r')
plt.show()

print(np.size(encoding))
while(True):
    img = ae.decode(encoding)
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    rand_idx = np.random.randint(np.size(encoding))
    encoding[0, rand_idx] = np.random.randint(2)
