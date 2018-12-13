from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # added no of classes
        self.no_classes = output_size  

    def sigmoid(self, x):
        """Compute sigmoid values for each sets of scores in x."""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """Compute relu values for each sets of scores in x."""
        return np.maximum(x, 0)
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x) , axis = -1 , keepdims = True)

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])   

    def L_i_vectorized(self, scores , y):
        """
        A faster half-vectorized implementation. half-vectorized
        refers to the fact that for a single example the implementation contains
        no for loops, but there is still one loop over the examples (outside this function)
        """
        delta = 1.0
        # compute the margins for all classes in one vector operation
        margins = np.maximum(0, scores - scores[y] + delta)
        # on y-th position scores[y] - scores[y] canceled and gave delta. We want
        # to ignore the y-th position and only consider margin on max wrong class
        margins[y] = 0
        loss_i = np.sum(margins)
        return loss_i

    
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        

        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        
        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # first layer activations
        z1 = np.dot(X, W1) + b1
        h1 = self.relu(z1)
        #second layer activations
        z2 = np.dot(h1, W2) + b2
        h2 = self.softmax(z2)
        scores = z2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
          return scores

        # Compute the loss
        loss = 0
        scores = h2
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # regularization loss
        reg_loss = reg * (np.sum(np.dot(W1, W1.T)) + np.sum(np.dot(W2, W2.T))) 
        # cross-entropy-loss
        loss =  - 1/N * np.sum(np.log10(scores)) 
        loss += reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        
        """
            l = -sum(p * log(h2)) / N
            dldh2 = - sum(1 / h2) / N
            dldz2 = dldh2 * dh2dz2 , dh2dz2 = h2 * (1- h2) -> derv of softmax
            dldw2 = dldz2 * dz2dw2 , dz2dw2 = h1
            dldb2 = dldz2 * dz2db2 , dz2db2 = 1
            dldh1 = dldz2 * dz2dh1 , dz2dh1 = w2
            dldz1 = dldh1 * dh1dz1 , dh1dz1 = 1(z1 > 0)
            dldw1 = dldz1 * dz1dw1 , dz1dw1 = x
            dldh1 = dldz1 * dz1dh1 , dz1dh1 = 1
        
        """
        dldh2 = - np.sum( self.get_one_hot(y, self.no_classes) / h2) / N
        #print("dldh2 :", np.shape(dldh2))
        dh2dz2 = h2 * (1- h2)
        #print("dh2dz2 :", np.shape(dh2dz2))
        dldz2 = dldh2 * dh2dz2
        #print("dldz2 :", np.shape(dldz2))
        dz2dw2 = h1
        #print("dz2dw2 :", np.shape(dz2dw2))
        dldw2 = np.dot(dz2dw2.T, dldz2)
        #print("dldw2 :", np.shape(dldw2))
        dz2db2 = 1
        #print("dz2db2 :", np.shape(dz2db2))
        dldb2 = dldz2 * dz2db2
        #print("dldb2 :", np.shape(dldb2))
        dz2dh1 = W2
        #print("dz2dh1 :", np.shape(dz2dh1))
        dldh1 = np.dot(dldz2, dz2dh1.T) 
        #print("dldh1 :", np.shape(dldh1))
        dh1dz1 = (z1 > 0)
        #print("dh1dz1 :", np.shape(dh1dz1))
        dldz1 = dldh1 * dh1dz1
        #print("dldz1 :", np.shape(dldz1))
        dz1dw1 = X
        #print("X :", np.shape(X))
        dldw1 = np.dot(dz1dw1.T, dldz1) 
        #print("dldw1 :", np.shape(dldw1))
        dz1dh1 = 1
        #print("dz1dh1 :", np.shape(dz1dh1))
        dldb1 = dldz1 * dz1dh1
        #print("W1", np.shape(W1))
        #print("W2", np.shape(W2))
        grads['W1'] = dldw1
        grads['W2'] = dldw2
        grads['b1'] = dldb1
        grads['b2'] = dldb2
        
          
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
          X_batch = None
          y_batch = None

          #########################################################################
          # TODO: Create a random minibatch of training data and labels, storing  #
          # them in X_batch and y_batch respectively.                             #
          #########################################################################
          pass
          #########################################################################
          #                             END OF YOUR CODE                          #
          #########################################################################

          # Compute loss and gradients using the current minibatch
          loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
          loss_history.append(loss)

          #########################################################################
          # TODO: Use the gradients in the grads dictionary to update the         #
          # parameters of the network (stored in the dictionary self.params)      #
          # using stochastic gradient descent. You'll need to use the gradients   #
          # stored in the grads dictionary defined above.                         #
          #########################################################################
          pass
          #########################################################################
          #                             END OF YOUR CODE                          #
          #########################################################################

          if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

          # Every epoch, check train and val accuracy and decay learning rate.
          if it % iterations_per_epoch == 0:
            # Check accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        pass
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred


