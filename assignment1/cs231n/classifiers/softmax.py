import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[1]
  num_class = W.shape[0]
  Y_hat = np.exp(np.dot(W, X))
  prob = Y_hat / np.sum(Y_hat, axis = 0)

  # C x N array, element(i,j)=1 if y[j]=i
  ground_truth = np.zeros_like(prob)
  ground_truth[tuple([y, range(len(y))])] = 1.0

  for i in xrange(num_data):
    for j in xrange(num_class):
      loss += -(ground_truth[j, i] * np.log(prob[j, i]))/num_data
      dW[j, :] += -(ground_truth[j, i] - prob[j, i])*(X[:,i]).transpose()/num_data
  loss += 0.5*reg*np.sum(np.sum(W**2, axis = 0)) # reg term
  dW += reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.

  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W


  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  (num_class, D) = W.shape
  num_data = X.shape[1]

  scores = np.dot(W, X)
  scores -= np.max(scores) # shift by log C to avoid numerical instability

  y_mat = np.zeros(shape = (num_class, num_data))
  y_mat[y, range(num_data)] = 1

  # matrix of all zeros except for a single wx + log C value in each column that corresponds to the
  # quantity we need to subtract from each row of scores
  correct_wx = np.multiply(y_mat, scores)

  # create a single row of the correct wx_y + log C values for each data point
  sums_wy = np.sum(correct_wx, axis=0) # sum over each column

  exp_scores = np.exp(scores)
  sums_exp = np.sum(exp_scores, axis=0) # sum over each column
  result = np.log(sums_exp)

  result -= sums_wy

  loss = np.sum(result)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= float(num_data)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  sum_exp_scores = np.sum(exp_scores, axis=0) # sum over columns
  sum_exp_scores = 1.0 / (sum_exp_scores + 1e-8)

  dW = exp_scores * sum_exp_scores
  dW = np.dot(dW, X.T)
  dW -= np.dot(y_mat, X.T)

  dW /= float(num_data)

  # Add regularization to the gradient
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
