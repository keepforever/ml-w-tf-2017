import tensorflow as tf

x = tf.constant([[1.,2.]])

neg_x = tf.negative(x)

# the arg 'log_device_placement=True' tells our code to print session info.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: 
    result = sess.run(neg_x)


print(result)

# This outputs info about which CPU/GPU devices are used 
# in the session for each operation
