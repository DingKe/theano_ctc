import theano
import theano.tensor as T

import numpy as np

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

y_true = T.imatrix()
y_pre = T.ftensor3()

# flat_labels
flat_labels = _make_flat(y_true)
f1 = theano.function([y_true], flat_labels)

input_lengths = theano.gradient.disconnected_grad(T.sum(T.ones_like(y_pre[:,:,0]), axis=1, dtype='int32'))
label_lengths = T.sum(y_true >= 0, axis=1, dtype='int32')
f2 = theano.function([y_pre, y_true], [input_lengths, label_lengths])


y = np.asarray([[0,1,2,-1], [3, 4,-1,-1]], dtype='int32')
yhat = np.asarray(np.random.rand(2,3,4), 'float32')

fl = f1(y)
il, ll = f2(yhat, y)

print y
print yhat
print fl
print il
print ll
