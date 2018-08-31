import tensorflow as tf
import numpy as np

raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg


avg_hist = tf.summary.scalar("running_average", update_avg) 
value_hist = tf.summary.scalar("incoming_values", curr_value)  
merged = tf.summary.merge_all()  
writer = tf.summary.FileWriter("./logs")  
init = tf.global_variables_initializer()


# sets up the primary loop and calls the update_avg
# operator on each iteration. Running the update_avg
# operator depends on the curr_value, which is fed using
#  the feed_dict argument.
with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run(
            [merged, update_avg],
            feed_dict={curr_value: raw_data[i]}
        )
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        print('summary_str', i, summary_str)
        writer.add_summary(summary_str, i)

#A Create a summary node for the averages 
#B Create a summary node for the values 
#C Merge the summaries to make it easier to run together 
#D Pass in the “logs” directory location to the writer 
#E Optional, but it allows you to visualize the computation 
# graph in TensorBoard #F Run the merged op and the update_avg 
# op at the same time #G Add the summary to the writer
