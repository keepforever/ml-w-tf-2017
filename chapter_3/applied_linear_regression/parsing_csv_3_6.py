import csv
import time
import pandas as pd


# crime14_freq = data_reader.read('../../../crimes_2014.csv', 1, '%d-%b-%y %H:%M:%S', 2014)
# freq = read('311.csv', 0, '%m/%d/%Y', 2014)

def read(filename, date_idx, date_parse, year, bucket=7):

    days_in_year = 365

    # Create initial frequency map
    freq = {}
    for period in range(0, int(days_in_year/bucket)):
        freq[period] = 0

    # Read data and aggregate crimes per day

    with open(filename, "rt", encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[date_idx] == '':
                continue
            t = time.strptime(row[date_idx], date_parse)
            if t.tm_year == year and t.tm_yday < (days_in_year-1):
                freq[int(t.tm_yday / bucket)] += 1

    return freq
# for printinng, need freq to do data science below
# if __name__ == '__main__':
#     freq = read('../../../311.csv', 0, '%m/%d/%Y', 2014)
#     print(freq)

# tring generate a continous model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#reproducable results
np.random.seed(100)

# freq = read('../../../311.csv', 0, '%m/%d/%Y', 2014)
#
# (pd.DataFrame.from_dict(data=freq, orient='index')
#    .to_csv('dict_file.csv', header=False))

with open("dict_file.csv") as f:
    reader = csv.reader(f)
    #don't actually need to skip header in this case
    # next(reader) # skip header
    freq = [r for r in reader]

print(freq, '\n \n')

x_dataset = []
y_dataset = []

for key_val_pair in freq:
    x_dataset.append(key_val_pair[0])
    y_dataset.append(key_val_pair[1])

# x_dataset = freq.keys()
# y_dataset = freq.values()
#
# print("x_dataset", x_dataset, '\n \n')
# print("y_dataset", y_dataset)

#helper method to split data
# def split_dataset(x_dataset, y_dataset, ratio):
#     arr = np.arange(x_dataset.size)
#     np.random.shuffle(arr)
#     num_train = int(ratio * x_dataset.size)
#     x_train = x_dataset[arr[0:num_train]]
#     y_train = y_dataset[arr[0:num_train]]
#     x_test = x_dataset[arr[num_train:x_dataset.size]]
#     y_test = y_dataset[arr[num_train:x_dataset.size]]
#     return x_train, x_test, y_train, y_test

# splt data into train/test 70/30 using helper func above
# (x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)
#
import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.30, random_state=42)

print('x_train: ','\n \n', x_train, '\n  \n')
print('y_train: ','\n \n', y_train)

#ok, now time to tensorflow

# hyper-params and constants
learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.
num_coeffs = 9

# set up input/output placeholder vars
X = tf.placeholder("float")
Y = tf.placeholder("float")

# define our model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        print(term)
        terms.append(term)
    return tf.add_n(terms)


# define regularization cost function
w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)),
                     tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))),
              2*len(x_train))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# set up the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for reg_lambda in x_dataset:
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
    final_cost = sess.run(cost, feed_dict={X: x_test, Y:y_test})
    print('reg lambda', reg_lambda)
    print('final cost', final_cost)

# close the session
sess.close()
