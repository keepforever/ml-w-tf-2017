import tensorflow as tf 

x = tf.constant([1.,2.])
neg_op = tf.negative(x)

with tf.Session() as sess:
    result = sess.run(neg_op)

print(result)

# every tensor object has an eval() operation to evaluate the 
# math ops that define it's value.