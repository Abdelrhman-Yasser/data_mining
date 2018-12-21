import numpy as np
from random import shuffle

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x) , axis = -1 , keepdims = True)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes]) 

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    return softmax_loss_vectorized(W, X, y, reg)



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N, D = X.shape
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    scores = softmax(scores)
    # regularization loss
    reg_loss = reg * np.sum(np.multiply(W, W))
    # cross-entropy-loss
    one_hot = get_one_hot(y, num_classes)
    # maximum loss = (-1/N) * sum over N samples ([0000 1 000] * [1/N 1/N ......])
    # which will be -1 * np.log(0.1) 
    loss =  - 1/N * np.sum( one_hot * np.log(scores) )
    loss += reg_loss
    # gradient
    dldz2 = scores - one_hot 
    dz2dw2 = X
    dW = 1/N * np.dot(dz2dw2.T, dldz2) + 2 * reg * W    
    return loss, dW

