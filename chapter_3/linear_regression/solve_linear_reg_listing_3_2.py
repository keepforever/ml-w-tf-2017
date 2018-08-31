import numpy as np  
import matplotlib.pyplot as plt   
import tensorflow as tf 

learning_rate = 0.01  
training_epochs = 100  

x_train = np.linspace(-1, 1, 101) 
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32) 



def model(X, w):
  return tf.multiply(X, w)       

w = tf.Variable(0.0, name="weights")

y_model = model(X, w) 
cost = tf.square(Y - y_model) 


train_op = tf.train.GradientDescentOptimizer(
    learning_rate
    ).minimize(cost)


sess = tf.Session() 
init = tf.global_variables_initializer() 
sess.run(init)  

for epoch in range(training_epochs):  
    for (x, y) in zip(x_train, y_train):  
        sess.run(train_op, feed_dict={X: x, Y: y})

w_val = sess.run(w) 

sess.close() 

plt.scatter(x_train, y_train)  
y_learned = x_train*w_val  
plt.plot(x_train, y_learned, 'r')  
plt.show()  

#A Import TensorFlow for the learning algorithm. 
# We'll need NumPy to set up the initial data.
#  And we'll use matplotlib to visualize our data.
#  #B Define some constants used by the learning algorithm.
#  There are called hyper-parameters. 
# #C Set up fake data that we will use to find a best fit line 
# #D Set up the input and output nodes as placeholders since 
# the value will be injected by x_train and y_train. 
# #E Define the model as y = w*x #F Set up the weights variable 
# #G Define the cost function 
# #H Define the operation that will be called on each 
# iteration of the learning algorithm 
# #I Set up a session and initialize all variables 
# #J Loop through the dataset multiple times 
# #K Loop through each item in the dataset 
# #L Update the model parameter(s) to try to minimize the
#  cost function #M Obtain the final parameter value 
# #N Close the session #O Plot the original data 
# #P Plot the best fit line
