import tensorflow as tf
# A Start the session in interactive mode so we 
# won’t need to pass around sess
sess = tf.InteractiveSession() 

# B Let’s say we have some raw data like this
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]

# C Create a Boolean variable called spike to
# detect a sudden increase in a series of numbers
spike = tf.Variable(False)
# D Because all variables must be initialized, initialize
# the variable by calling run() on its initializer
spike.initializer.run()

# E Loop through the data (skipping the first element) 
# and update the spike variable when there is a significant increase
for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        #F To update a variable, assign it a new value using 
        # tf.assign(<var name>, <new value>). Evaluate it to see the change.
        updater = tf.assign(spike, True)
        updater.eval()
    else: 
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())
#G Remember to close the session after it will no longer be used
sess.close()
