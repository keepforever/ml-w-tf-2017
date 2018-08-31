from scipy.misc import imread, imresize

# A 
small_gray_image = imresize(gray_image, 1. / 8.) 
# B 
x = small_gray_image.flatten()   
#C
gray_image = imread(filepath, True) 
