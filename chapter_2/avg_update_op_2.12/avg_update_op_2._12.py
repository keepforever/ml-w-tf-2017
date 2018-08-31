# A alpha is a tf.constant, curr_value is a placeholder,
#  and prev_avg is a variable. 

# by defining the interface first, it forces us to implement 
# the peripheral setup code to satisfy the interface

# create normal dist 
import tensorflow as tf 
import numpy as np

#A Create a vector of 100 numbers, raw_data, 
# with a mean of 10 and standard deviation of 1
raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)  

prev_avg = tf.Variable(0.)  
update_avg = alpha * curr_value + (1 - alpha) * prev_avg

init = tf.global_variables_initializer()

# sets up the primary loop and calls the update_avg 
# operator on each iteration. Running the update_avg 
# operator depends on the curr_value, which is fed using
#  the feed_dict argument.
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(raw_data)):
        curr_avg = sess.run(
            update_avg, 
            feed_dict={curr_value: raw_data[i]}
        )
        sess.run(tf.assign(prev_avg, curr_avg)) 
        print(raw_data[i], curr_avg)


#A Create a vector of 100 numbers with a mean of 10 and
#  standard deviation of 1 #B Define alpha as a constant 
# #C A placeholder is just like a variable, but the value
#  is injected from the session  #D Initialize the previous
#  average to zero #E Loop through the data one-by-one to 
# update the average


# Visualization using TensorBoard is usually a two-step process.  
# 1. First, you must pick out which nodes you really care 
# about measuring by annotating them with a summary op.  
# 2. Then, call add_summary on them to queue up data
#  to be written to disk 

#img = tf.placeholder(tf.float32, [None, None, None, 3]) 
#cost = tf.reduce_sum(...) 
#my_img_summary = tf.summary.image("img", img)
#my_cost_summary = tf.summary.scalar("cost", cost)
