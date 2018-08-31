import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 40

trX = np.linspace(-1, 1, 101)

# C
num_coeffs = 6
trY_coeffs = [1,2,3,4,5,6]
trY = 0

#C Set up raw output data based on a degree 5 polynomial
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
# D add some noise
trY += np.random.randn(*trX.shape) * 1.5

plt.scatter(trX, trY)  
# E 
plt.show()
# F
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


# H
w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)  

# I
cost = (tf.pow(Y - y_model, 2)) 
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# J
sess = tf.Session() 
init = tf.global_variables_initializer()  
sess.run(init)  
# J still...
for epoch in range(training_epochs):
    for (x, y) in zip(trX, trY):
        print('x, y: ', x, y)
        sess.run(train_op, feed_dict={X: x, Y: y})

w_val = sess.run(w)

sess.close()

# L 
trY2 = 0  
for i in range(num_coeffs):  
    trY2 += w_val[i] * np.power(trX, i)

plt.scatter(trX, trY) 
plt.plot(trX, trY2, 'r') 
plt.show() 


#A Import the relevant libraries and
#  initialize the hyper-parameters 
#B Set up some fake raw input data 
#C Set up raw output data based on a degree 5 polynomial
#D Add some noise 
#E Show a scatter plot of the raw data 
#F Define the nodes to hold values for input/output pairs 
#G Define our polynomial model 
#H Set up the parameter vector to all zeros 
#I Define the cost function just as before
#J Set up the session and run the learning 
# algorithm just as before 
# #K Close the session when done 
# #L Plot the result
