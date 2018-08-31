import tensorflow as tf 
import numpy as np

# Different ways to represent tensors

# A python list
m1 = [
    [1.0, 2.0],
    [3.0, 4.0]
]

# a numpy array
m2 = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
], dtype=np.float32)

# a tensorflow tensor 
m3 = tf.constant([[1.0, 2.0],
                  [3.0, 4.0]])

print('First print')
print(type(m1))
print(type(m2))
print(type(m3))

#convert above types to single type: tf.tensor
t1 = tf.convert_to_tensor(m1, dtype=tf.float32) 
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)   
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print('Second print')
print(type(t1)) 
print(type(t2))   
print(type(t3))

# tf.convert_to_tensor() is a function we can use anywhere to
# verify we're dealing in tensors.
