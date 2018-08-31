import tensorflow as tf
from os import listdir
from os.path import isfile, join

# A Store filenames that match a pattern #
filenames = ["audio_dataset/" + f for f in listdir("audio_dataset") if isfile(
    join("audio_dataset", f)) and f.endswith('.wav')]
count_num_files = tf.size(filenames)

# B Set up an pipeline for retrieving filenames randomly #
filename_queue = tf.train.string_input_producer(filenames)

# C Natively read a file in TensorFlow #
reader = tf.WholeFileReader()

# D Run the reader to extract file data #
filename, file_contents = reader.read(filename_queue)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # E Count the number of files #
    num_files = sess.run(count_num_files)
    # F Initialize threads for the filename queue #
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #something from stack overflow to tidy up a function call and stop warning
    # from printing. 
    coord.request_stop()
    coord.join(threads)
    # G Loop through the data one by one
    for i in range(num_files):
        audio_file = sess.run(filename)
        print(audio_file)
