import theano
import theano.tensor as T

import numpy as np

from theano_ctc.cpu_ctc import cpu_ctc_cost
from theano_ctc.gpu_ctc import gpu_ctc_cost
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import gpu_from_host
import os
from ctypes import cdll

cdll.LoadLibrary(os.path.join(os.environ["CTC_LIB"], "build", "libwarpctc.so"))

def ctc_cost(acts, input_lengths, flat_labels, label_lengths):
  # This should be properly integrated into the theano optimization catalog.
  # Until then, this forces the choice based on device configuration.
  if theano.config.device.startswith("gpu") or theano.sandbox.cuda.cuda_enabled:
    if not isinstance(acts.type, CudaNdarrayType): # if not already on the device
      acts = gpu_from_host(acts)  # this should get optimized away
    return gpu_ctc_cost(acts, input_lengths, flat_labels, label_lengths)
  else:
    return cpu_ctc_cost(acts, input_lengths, flat_labels, label_lengths)
    
def _make_flat(y_true):
    flat_labels = T.flatten(y_true)

    def step(i, j, flat_labels):
        val = flat_labels[i]

        cur = T.switch(val >= 0, val, flat_labels[j])
        sub_tensor = flat_labels[j]
        flat_labels = T.set_subtensor(sub_tensor, cur)

        j = T.switch(val >= 0, j + 1, j)

        return j, flat_labels

    results, _ = theano.scan(step,
                    sequences=T.arange(0, flat_labels.size, 1),
                    outputs_info=[T.as_tensor_variable(np.asarray(0, flat_labels.dtype)),
                                  flat_labels])

    return results[-1][-1]

def ctc_objective(y_true, y_pre):
    '''Convenient wrapper for ctc_cost, assuming all sentences have the same length
        Arguments:
            y_true: nb_minibatch by max_label_num, padded by -1
            y_pre: acts
    '''
    acts = y_pre
    # disconnect gradient to not to confuse graph building
    input_lengths = theano.gradient.disconnected_grad(T.sum(T.ones_like(y_pre[:,:,0]), axis=1, dtype='int32'))
    label_lengths = T.sum(y_true >= 0, axis=1, dtype='int32')
    flat_labels = _make_flat(y_true)
    return ctc_cost(acts, input_lengths, flat_labels, label_lengths)
