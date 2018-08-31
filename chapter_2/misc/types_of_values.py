# A session not only runs a graph operation, but can also take 
# placeholders, variables, and constants as input. We’ve used 
# constants so far, but in later sections we’ll start using 
# variables and placeholders. Here’s a quick overview of 
# these three types of values. 

# • Placeholder A value that is unassigned, but will be initialized 
# by the session wherever it is run. Typically, placeholders are the
#  input and output of your model. 
 
# • Variable A value that can change, such as parameters of a machine 
# learning model. Variables must me initialized by the session before
#  they are used. 

# • Constant A value that does not change, such as hyper-parameters
#  or settings.
