#A Import relevant libraries 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#B Set the hyper-parameters
learning_rate = 0.01
training_epochs = 1000

#C Define a helper function to calculate the sigmoid function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#D Initialize fake data
x1 = np.random.normal(-4, 2, 1000)
x2 = np.random.normal(4, 2, 1000)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))

#E Visualize the data
plt.scatter(xs, ys)
#F Define the input/output placeholders
X = tf.placeholder(tf.float32, shape=(None,), name="x")
Y = tf.placeholder(tf.float32, shape=(None,), name="y")
#G Define the parameter node
w = tf.Variable([0., 0.], name="parameter", trainable=True)
#H Define the model using TensorFlow’s sigmoid function
y_model = tf.sigmoid(-(w[1] * X + w[0]))
#I Define the cross-entropy loss function
cost = tf.reduce_mean(-tf.log(y_model * Y + (1 - y_model) * (1 - Y)))
#J Define the minimizer to use 
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#K Open a session and define all variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #L Define a variable to track of the previous error
    prev_err = 0
    #M Iterate until convergence or until maximum number of epochs reached
    for epoch in range(training_epochs):
        #N Computer the cost as well as update the learning parameters
        err, _ = sess.run([cost, train_op], {X: xs, Y: ys})
        print(epoch, err)
        #O Check for convergence – if we’re changing by < .01% per iteration, we’re done
        if abs(prev_err - err) < 0.0001:
            break
        #P Update the previous error value
        prev_err = err
    #Q Obtain the learned parameter value
    w_val = sess.run(w, {X: xs, Y: ys})

#R Plot the learned sigmoid function
all_xs = np.linspace(-10, 10, 100)
plt.plot(all_xs, sigmoid(all_xs * w_val[1] + w_val[0]))
plt.show()


