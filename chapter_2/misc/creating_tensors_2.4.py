import tensorflow as tf

# shape 1 x 2
m1 = tf.constant([[1., 2.]]) 

# shape 2 x 1
m2 = tf.constant([
    [1],  
    [2]
]) 

# shape = 2 x 3 x 2
m3 = tf.constant([
    [
        [1, 2], 
        [3, 4],
        [5, 6]
    ],
    [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
]) 


print(m1) 
print(m2)   
print(m3)

# OUTPUT: 

# Tensor("Const:0", shape=(1, 2), dtype=float32)
# Tensor("Const_1:0", shape=(2, 1), dtype=int32)
# Tensor("Const_2:0", shape=(2, 3, 2), dtype=int32)

# since we didn't specify a name, "Const:0" is automatically generated.
