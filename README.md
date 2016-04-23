Add cost_objective() for easy integrated into deep learning framework such as [keras](http://keras.io).

## Usage
Use the ctc_objective() function like any other Theano Op:

    from theano_ctc import ctc_objective
    import theano.tensor as T

    costs = ctc_objective(labels, acts)
    grad = T.grad(T.mean(costs), acts)

ctc_objective assumes that all sentences have the same length of acts (by sorting and padding) and labels (by padding -1).
See the test_ctc3.py for a working example.

original README

# theano_ctc 
Theano bindings for Baidu's CTC library. Supports CPU and GPU computation.

## Installation
Clone and compile [warp-ctc](https://github.com/baidu-research/warp-ctc).

Clone and install theano_ctc:

    git clone https://github.com/mcf06/theano_ctc.git
    cd theano_ctc
    pip install .

Configure environment:

    export CTC_LIB=/path/to/warp-ctc

such that $CTC_LIB/build/libwarpctc.so exists.

Read the [Torch Tutorial](https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md) and try out the Theano version:

    THEANO_FLAGS=device=gpu0 python tests/test_ctc.py

## Usage
Use the ctc_cost() function like any other Theano Op:

    from theano_ctc import ctc_cost
    import theano.tensor as T

    costs = ctc_cost(acts, actT, labels, labelT)
    grad = T.grad(T.mean(costs), acts)

See the test scripts for a working example.

## Status

This would benefit from integration into Theano, particularly with respect to the GPU optimizations. Currently, ctc_cost always uses the GPU implementation if a GPU device is enabled.
