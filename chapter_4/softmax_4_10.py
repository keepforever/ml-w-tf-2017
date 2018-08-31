#  Unlike the sigmoid function in logistic regression, here we will use the softmax
#  function provided by the TensorFlow library. The softmax function is similar to
#  the max function, which simply outputs the maximum value from a list of numbers.
#  It’s called softmax because it’s a “soft” or “smooth” approximation of the max
#  function, which is not smooth or continuous (and that’s bad). Continuous and
#  smooth functions facilitate learning the correct weights of a neural network 
# by back-propagation.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# A Define hyper-parameters  #
learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 100
# fake data, normal dist, two independant vars, x1, x2 for each label.
x1_label0 = np.random.normal(1, 1, (100, 1))
x2_label0 = np.random.normal(1, 1, (100, 1))
x1_label1 = np.random.normal(5, 1, (100, 1))
x2_label1 = np.random.normal(4, 1, (100, 1))
x1_label2 = np.random.normal(8, 1, (100, 1))
x2_label2 = np.random.normal(0, 1, (100, 1))
# plot the data
plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
plt.show()

# np.hstack((a,b)) explained: 
# a = [0,1,2], b = [3,4,5]
# np.hstack((a,b)) => [0,1,2,3,4,5]
xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
# np.vstack((a,b)) explained:
# a = [0,1,2], b = [3,4,5]
# np.vstack((a,b)) => [
#   [0,1,2], 
#   [3,4,5]
# ]
xs = np.vstack((xs_label0, xs_label1, xs_label2))
print('xs.shape[0]', '\n ', xs.shape[0], '\n')
# create len(x1_label0) 
labels = np.matrix([[1., 0., 0.]] * len(x1_label0) + [[0., 1., 0.]]
                   * len(x1_label1) + [[0., 0., 1.]] * len(x1_label2))

arr = np.arange(xs.shape[0])
print('arr', '\n ', arr, '\n')
# shuffle data and match up labels
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]

test_x1_label0 = np.random.normal(1, 1, (10, 1))
test_x2_label0 = np.random.normal(1, 1, (10, 1))
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))

test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix([[1., 0., 0.]] * 10 +
                        [[0., 1., 0.]] * 10 + [[0., 0., 1.]] * 10)

# print('xs', '\n', xs)
train_size, num_features = xs.shape

print('num_features', num_features)

# B Define the input/output placeholder nodes #
X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

# C Define the model parameters #
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
# D Design the softmax model #
y_model = tf.nn.softmax(tf.matmul(X, W) + b)

# E Set up the learning algorithm #
cost = -tf.reduce_sum(Y * tf.log(y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# F Define an op to measure success rate
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# H Open a new session and initialize all variables #
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # I Loop only enough times to complete a single pass through the dataset #
    for step in range(training_epochs * train_size // batch_size):
        # J Retrieve a subset of the dataset corresponding to the current batch #
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]
        # K Run the optimizer on this batch #
        err, _ = sess.run([cost, train_op], feed_dict={
                          X: batch_xs, Y: batch_labels})
        # L Print on-going results #
        # print(step, err)

    # M Print final learned parameters #
    W_val = sess.run(W)
    # W_val is a 2x3 matrix of constants because 2_features x 3_labels
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    # N Print success rate
    print ("accuracy", accuracy.eval(feed_dict={X: test_xs, Y: test_labels}))


