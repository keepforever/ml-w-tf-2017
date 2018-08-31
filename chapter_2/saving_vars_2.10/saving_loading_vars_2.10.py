# something is wrong with this code from the book
# the saver gives an error. 

import tensorflow as tf
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]

spikes = tf.Variable([False] * len(raw_data), name='spikes')
print(spikes)
spikes.initializer.run()

saver = tf.train.Saver()

# E Loop through the data (skipping the first element)
# and update the spike variable when there is a significant increase
for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

# we create a tmp directory because the save operation makes 
# more than one file. 
save_path = saver.save(sess, "./spikes/spikes.ckpt")
# #G Remember to close the session after it will no longer be used
print("spikes data saved in file: %s" % save_path)
sess.close()
