import tensorflow as tf

# A Store filenames that match a pattern #
filenames = tf.train.match_filenames_once('./audio_dataset/*.wav')
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
    # G Loop through the data one by one
    for i in range(num_files):
        audio_file = sess.run(filename)  
        print(audio_file)   
        
