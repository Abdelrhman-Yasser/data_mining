import numpy as np
from random import shuffle

# see that https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss
# to understand more on loss function and gradient of linear svm
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # loss of this example point with the class j
        loss += margin
        # gradient of loss function w.r.t the y[i] at the sample i
        dW[:, j] += X[i]
        dW[:, y[i]] += (-1) * X[i]
      else:
      	# add zero to the gradient
      	pass

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # average the total gradient w.r.t y[i] for all the samples
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient.
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # scores for all samples over all classes
  scores = X.dot(W)
  # correct class score for all samples
  # so indexing by all the samples in the rows, and the columns will be the
  # correct class index
  correct_class_scores = scores[range(0, X.shape[0]), y]
  # calculte the margin for all the samples with all wrong classes
  # margins of dimensions (N x D)
  correct_class_scores = correct_class_scores.reshape(X.shape[0], -1)
  margins = scores - correct_class_scores + 1
  # set all negative margins to be zero
  margins[margins < 0] = 0
  margins[range(0, X.shape[0]), y] = 0
  # calculate loss term using margins
  loss = np.sum(margins)
  loss /= X.shape[0]
  # calculate loss term using regularization 
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  multipliers = np.zeros(margins.shape)
  multipliers[margins > 0] = 1
  multipliers[np.arange(X.shape[0]), y] = -1 * np.sum(multipliers, axis=1)
  dW = X.T.dot(multipliers)
  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
