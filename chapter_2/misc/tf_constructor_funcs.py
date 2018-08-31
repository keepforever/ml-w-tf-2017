# init a 500-by-500 tensor with all elements equal 0.5
import tensorflow as tf 

t_500 = tf.ones([500, 500]) * 0.5

print(t_500)