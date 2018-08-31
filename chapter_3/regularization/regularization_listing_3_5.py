#page 67 in pdf

# regularization introduces a penalty, lamda, to the
# cost funtion.  The penalty's affect
# increases with the L2 norm distance from the target.
# The intension is to avoid overfitting
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#reproducable results
np.random.seed(100)
# hyper params
learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.

#helper method to split data
def split_dataset(x_dataset, y_dataset, ratio):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_dataset.size)
    x_train = x_dataset[arr[0:num_train]]
    y_train = y_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test

# create fake dataset y = x^2
x_dataset = np.linspace(-1, 1, 100)

num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

# splt data into train/test 70/30 using helper func above
(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)

# set up input/output placeholder vars
X = tf.placeholder("float")
Y = tf.placeholder("float")

# define our model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

# define regularization cost function
w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)),
                     tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))),
              2*x_train.size)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# set up the session

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for reg_lambda in np.linspace(0,1,100):
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
    final_cost = sess.run(cost, feed_dict={X: x_test, Y:y_test})
    print('reg lambda', reg_lambda)
    print('final cost', final_cost)

# close the session
sess.close()

# If we plot the corresponding output per each regularization
# parameter from listing 3.5, we can see how the curve changes as
# λ increases. When λ is 0, the algorithm favors using the higher
# order terms to fit the data. As we start penalizing parameters 
# with a high L2 norm, the cost decreases, indicating that we
# are recovering from overfitting.
