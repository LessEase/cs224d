##
# Miscellaneous helper functions
##

from numpy import random
from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    eps = sqrt(6.0/(m+n))
    A0 = random.uniform(-eps, eps, size=(m, n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0