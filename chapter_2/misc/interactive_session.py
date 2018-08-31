import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant([[1.,2.]])

neg_x = tf.negative(x)

result = neg_x.eval()

print(result)

sess.close()