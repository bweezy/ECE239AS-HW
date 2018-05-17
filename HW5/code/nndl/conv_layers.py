import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  npad = ((0,0), (0,0), (pad,pad), (pad,pad))
  x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
  H_prime = 1 + (H + 2*pad - HH)/stride
  W_prime = 1 + (W + 2*pad - WW)/stride

  out = np.empty((N,F,H_prime, W_prime))
  
  for n in np.arange(0,N):
    for i in np.arange(0, F):
      for j in np.arange(0, H_prime):
        for k in np.arange(0, W_prime):
          out[n,i,j,k] = np.sum(w[i, :, :, :] * x_padded[n, :, j*stride:j*stride+HH, k*stride:k*stride+WW]) + b[i]



  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dx = np.zeros_like(x)
  dxpad = np.zeros_like(xpad)
  F, _, HH, WW = w.shape

  for n in np.arange(0,N):
    for f in np.arange(0, F):
      for j in np.arange(0, out_height):
        for k in np.arange(0, out_width):
          dw[f] += xpad[n, :, j*stride:j*stride+HH, k*stride:k*stride+WW]*dout[n, f, j, k]
          db[f] += dout[n,f,j,k]
          dxpad[n, :, j*stride:j*stride+HH, k*stride:k*stride+WW] += w[f]*dout[n,f,j,k]

  dx[:] = dxpad[:,:,pad:-pad,pad:-pad]





  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape

  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  h_prime = (H - ph)/stride + 1
  w_prime = (W - pw)/stride + 1

  out = np.empty((N,C,h_prime, w_prime))

  for n in np.arange(0,N):
    for c in np.arange(0,C):
      for h in np.arange(0,h_prime):
        for w in np.arange(0,w_prime):
          out[n,c,h,w] = np.max(x[n,c,h*stride:h*stride+ph,w*stride:w*stride+pw])


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  h_prime, w_prime = dout.shape[-2:]
  dx = np.zeros_like(x)

  for n in np.arange(0,N):
    for c in np.arange(0,C):
      for h in np.arange(0,h_prime):
        for w in np.arange(0,w_prime):
          idx = np.unravel_index( np.argmax( x[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]),
              (pool_height, pool_width))
          dx[n,c,h*stride+idx[0],w*stride+idx[1]] = dout[n,c,h,w]



  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #

  x_transpose = x.transpose((0,2,3,1))
  x_reshaped = x_transpose.reshape((-1,x.shape[1]))

  out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)

  out = out.reshape(*x_transpose.shape).transpose((0,3,1,2))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #

  dout_t = dout.transpose(0,2,3,1)
  dout_r = dout_t.reshape(-1, dout.shape[1])

  dx, dgamma, dbeta = batchnorm_backward(dout_r, cache)

  dx = dx.reshape(*dout_t.shape).transpose((0,3,1,2))

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta