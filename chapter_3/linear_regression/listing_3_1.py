# hilighting shows file has an error but it runs. 
import numpy as np
import matplotlib.pyplot as plt


x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

plt.scatter(x_train, y_train) 
plt.show() 


# A Import NumPy to help generate initial raw data
# B Use matplotlib to visualize the data 
# C The input values are 101 evenly spaced numbers
#  between -1 and 1 
# D The output values are proportional to the input
# but with added noise 
# E Use matplotlib’s function to generate a
#  scatter plot of the data


# NOW TO TRY AND FIT THE RANDOM DATA

# At the very least, you need to provide TensorFlow 
# with a score for each candidate parameter it tries. 

# Each step of looping through all your data to update 
# the parameters is called an epoch.

