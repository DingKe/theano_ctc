# This test is designed to mirror the tutorial in Torch, found here:
#   https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# This test computes the gradient with respect to the weights.

import numpy as np
import theano
import theano.tensor as T

theano.sandbox.cuda.use('gpu0')

from theano_ctc import ctc_objective

from theano.printing import debugprint as dprint

import sys

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  broadcastShape = x.shape[0:2] + (1,)
  e_x = np.exp(x - np.max(x, axis=2).reshape(broadcastShape))
  return e_x / e_x.sum(axis=2).reshape(broadcastShape)

targetN = 5
batchSize = 1
seqLen = 1

# time, batchSize, inputLayerSize
inputs = np.random.rand(10, 3, 6).astype(np.float32)

# weight matrix: inputLayerSize x outputLayerSize
weights = np.asarray([[1, 0, 0, 0, 0], \
                      [0, 1, 0, 0, 0], \
                      [0, 0, 1, 0, 0], \
                      [0, 0, 0, 1, 0], \
                      [0, 0, 0, 0, 1], \
                      [1, 1, 1, 1, 1]], dtype=np.float32)

# time, batchSize, outputLayerSize
acts = np.dot(inputs, weights)

print "Activations"
print acts
print
print "Softmax outputs"
print softmax(acts)
print

# labels for each sequence, padded
labels = np.asarray([[1, -1],
                     [3, 3], 
                     [2, 3]], dtype=np.int32)

# Symbolic equivalents
tsInputs = theano.shared(inputs, name="inputs")
tsWeights = theano.shared(weights, name="weights")
tsActs = T.dot(tsInputs, tsWeights)
tsLabels = theano.shared(labels, "labels")

print "tsActs:"
dprint(tsActs)
print

# CTC cost
tCost = ctc_objective(tsLabels, tsActs)

print "Symbolic CTC cost:"
dprint(tCost)
print "\n"

# Gradient of CTC cost
tGrad = T.grad(T.mean(tCost), tsWeights)

print "Symbolic gradient of CTC cost:"
dprint(tGrad)
print "\n"

f = theano.function([], [tCost, tGrad])
print "Theano function to calculate costs and gradient of mean(costs):"
dprint(f)
print

cost, grad = f()
print "cost:"
print cost
print
print "gradient of average ctc_cost with respect to weights:"
print np.asarray(grad)
