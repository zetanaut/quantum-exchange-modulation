"""Measured-feature comparators with fully specified, NumPy-only training."""
from __future__ import annotations
import numpy as np


def metrics(y, pred):
    e = np.asarray(pred) - y
    v = np.mean((y-y.mean())**2)
    return {'mse': float(np.mean(e**2)), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1-np.mean(e**2)/v) if v else None,
            'max_abs_error': float(np.max(np.abs(e)))}


def quad_context(c):
    return np.column_stack([np.ones(len(c)), c,
        *[c[:, i]*c[:, j] for i in range(3) for j in range(i,3)]])


class FiLMRegressor:
    """tanh(rW+b) -> FiLM(gamma(c),beta(c)) -> tanh -> affine output.

    Feature estimates are available classically. In exact training this is an
    ideal-expectation comparator; finite shots are assessed separately.
    """
    def __init__(self, n_features=15, width=12):
        self.n_features, self.width = n_features, width
        self.shapes = [(n_features, width), (width,), (3, width),
                       (3, width), (width,), (1,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def unpack(self, p):
        off, values = 0, []
        for shape, size in zip(self.shapes, self.sizes):
            values.append(p[off:off+size].reshape(shape))
            off += size
        return values

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 0.5, self.shapes[0])
        bh = rng.normal(0, .1, self.shapes[1])
        Wg = rng.normal(0, .1, self.shapes[2])
        Wb = rng.normal(0, .1, self.shapes[3])
        wo = rng.normal(0, .15, self.shapes[4])
        bo = np.array([.35])
        return np.concatenate([x.ravel() for x in (W,bh,Wg,Wb,wo,bo)])

    def predict(self, p, r, c):
        W,bh,Wg,Wb,wo,bo = self.unpack(p)
        h = np.tanh(r@W+bh)
        z = np.tanh(h*(1+c@Wg)+c@Wb)
        return z@wo+bo[0]

    def value_gradient(self, p, r, c, y, l2=1e-5):
        W,bh,Wg,Wb,wo,bo = self.unpack(p)
        h = np.tanh(r@W+bh)
        gamma = 1+c@Wg
        z = np.tanh(h*gamma+c@Wb)
        pred = z@wo+bo[0]
        residual = pred-y
        gy = 2*residual/len(y)
        gm = (gy[:,None]*wo)*(1-z*z)
        gp = gm*gamma*(1-h*h)
        pieces = [r.T@gp, gp.sum(axis=0), c.T@(gm*h),
                  c.T@gm, z.T@gy, np.array([gy.sum()])]
        grad = np.concatenate([g.ravel() for g in pieces]) + 2*l2*p
        return float(np.mean(residual**2)+l2*np.dot(p,p)), grad


def squared_distances(x, y):
    return np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None,:]-2*x@y.T,0)
