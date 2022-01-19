import numpy as np
from numpy.testing import assert_allclose
from tests.sol import *

def test_sigmoid(f, x):
    assert_allclose(f(x), sol_sigmoid(x))
    
def test_loss(logreg, x, y):
    assert_allclose(logreg.loss(x, y), sol_loss(x, y))

def test_predict(logreg, x):
    assert_allclose(logreg.predict(x), sol_predict(logreg, x))

    
def test_forward_pass(logreg, x, y):
    pred = logreg._forward_pass(x, y)
    gt = sol_forward_pass(logreg, x, y)
    assert len(pred) == len(gt)
    for i, j in zip(pred, gt):
        assert_allclose(i, j)
    
    return pred


def test_backward_pass(logreg, o, x, y):
    pred = logreg._backward_pass(o, x, y)
    gt = sol_backward_pass(o, x, y)
    assert len(pred) == len(gt)
    for i, j in zip(pred, gt):
        assert_allclose(i, j)
    
