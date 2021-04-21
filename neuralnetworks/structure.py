import numpy as np

def create_weight_matrix(nrows, ncols):
    return get_scaled_normal(nrows, ncols)

def create_bias_vector(length):
    return get_scaled_normal(length)


'''
returns a matrix containing a normal distribution
of size given by the inputs (cols default to 1) and values
between -+ 1/the product of the number of cells in the matrix.
Values closer to 0 are the peak of the normal distribution. 
'''
def get_scaled_normal(nrows, ncols=1):
    return np.random.normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))


# Leaky Rectified Linear Unit - Yeah, I don't know either.
# As a function, it moves -ve values closer to 0 and leaves =ve vals unchaged.
def leaky_relu(x, leaky_param=0.1):
    return np.maximum(x, x * leaky_param)