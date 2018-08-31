# in this file we import the autoencoder from autoencoder.py and use it on 
# arbitrary data

from autoencoder import Autoencoder
from sklearn import datasets

hidden_dim = 1

data = datasets.load_iris().data

input_dim = len(data[0])

ae = Autoencoder(input_dim, hidden_dim)

ae.train(data)

ae.test([[8, 4, 6, 2]])
