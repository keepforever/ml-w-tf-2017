  # To retrieve this data, you can use the restore
  #  function from the saver op, as demonstrated in listing 2.11

import tensorflow as tf
sess = tf.InteractiveSession()


spikes = tf.Variable([False]*8, name="spikes")
# DO NOT need to initialize var with initializer.run()
saver = tf.train.Saver()

saver.restore(sess, "./spikes/spikes.ckpt")

print(spikes.eval())

sess.close()

#A Create a variable of the same size and name as the saved 
# data #B You no longer need to initialize this variable because 
# it will be directly loaded #C Create the saver op to restore 
# saved data #D Restore data from the “spikes.ckpt” file #E Print
#  the loaded dat
