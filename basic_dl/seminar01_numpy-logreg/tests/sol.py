import numpy as np


def sol_sigmoid(x):
    return 1./ (1. + np.exp(-1.*x))

def sol_loss(x, y):
    pred = x
    pred = np.maximum(pred, 1e-5)
    pred = np.minimum(pred, 1.-1e-5)
    return (-1. * y * np.log(pred) + -1.*(1.-y)*np.log(1.-pred)).mean(axis=-1)

def sol_predict(logreg, x):
    return logreg.sigmoid(x.dot(logreg.w) + logreg.b)

def sol_forward_pass(logreg, X, y):
    z = X.dot(logreg.w) + logreg.b
    o = logreg.sigmoid(z)
    loss = logreg.loss(o, y).mean()
    return z, o, loss


def sol_backward_pass(o, X, y):
    dz = o - y
    dw = X.T.dot(dz) / X.shape[0]
    db = dz.mean(axis=0)
    return dz, dw, db
    