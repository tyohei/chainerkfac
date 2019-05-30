from chainer.backends import cuda

import cupy
import numpy as np

import warnings

from chainer.functions import cast
from chainer.functions.normalization.batch_normalization import BatchNormalization  # NOQA
from chainer.functions.normalization.batch_normalization import batch_normalization  # NOQA
from chainerkfac.optimizers.fisher_block import FisherBlock


def _compare_instance(a, b):
    if (a is None or b is None) or (a is not b):
        return False
    else:
        return True


def _check_nan(kfgrads, gamma, beta):
    xp = cuda.get_array_module(kfgrads)
    if xp.any(xp.isnan(kfgrads)):
        warnings.warn(
            'Detected NaN in the kfgrads of the batch normalization layer'
            'Substitue ``grad`` as ``kfgrad`` to avoid crashing')
        if gamma is not None:
            gamma.kfgrad = gamma.grad
        if beta is not None:
            beta.kfgrad = beta.grad
        return True
    else:
        return False


class FisherBlockBatchNormalization(FisherBlock):

    def __init__(self, *args, **kwargs):
        self._F = None
        super(FisherBlockBatchNormalization, self).__init__(*args, **kwargs)

    @property
    def funcclass(self):
        return BatchNormalization

    @property
    def cov_forward(self):
        return None  # cov forward doesn't exist

    @property
    def cov_backward(self):
        return self.covs[0]

    @property
    def inv_forward(self):
        return None  # inv forward doesn't exist

    @property
    def inv_backward(self):
        return self.invs[0]

    def is_mine(self, func, in_data, out_grad_data=None):
        if not isinstance(func, self.funcclass):
            return False
        if not _compare_instance(in_data[1], self.link.gamma.data) and \
                not _compare_instance(in_data[2], self.link.beta.data):
            return False
        return True

    def forward_postprocess(self, func, in_data):
        # Batch Normalization layer cannot compute F in forward
        pass

    def backward_preprocess(self, func, in_data, out_grad_data):
        self._F = self.compute_F(in_data, out_grad_data)
        self.covs = [self._F]

    def compute_F(self, in_data, out_grad_data):
        x = in_data[0]
        gy = out_grad_data[0]
        ndim = len(x.shape)
        if ndim not in (2, 4):
            raise RuntimeError(
                'len(x.shape) must be 2 or 4, not {}.'.format(ndim))

        xp = cuda.get_array_module(x)
        n = x.shape[0]
        gy_scale = n
        if self._loss_scale is not None:
            gy_scale *= 1.0 / self._loss_scale

        # Re-compute BN forward with gamma=1 and beta=0
        avg_mean = self.link.avg_mean
        _gamma = xp.ones(avg_mean.shape, dtype=x.dtype)
        _beta = xp.zeros(avg_mean.shape, dtype=x.dtype)
        h = batch_normalization(x, _gamma, _beta, eps=self.link.eps).data

        if ndim == 2:
            gy = gy_scale * gy
            gyh = gy * h
        elif ndim == 4:
            # data layout of gy: NCHW
            h = h.transpose(0, 2, 3, 1)
            gy = gy.transpose(0, 2, 3, 1)

            # data layout of gy: NHWC
            gy = gy * gy_scale  # copy
            gyh = gy * h

            gyh = gyh.sum(axis=(1, 2))
            gy = gy.sum(axis=(1, 2))
            # data layout of gy: NC

        if self.link.beta is None:
            grad = gyh
        elif self.link.gamma is None:
            grad = gy
        else:
            grad = xp.hstack((gyh, gy))

        if self.diagonalize:
            if grad.dtype == xp.float16:
                grad = cast(grad, xp.float32).data
            F = xp.diag((grad * grad).mean(axis=0))
        else:
            F_scale = 1 / n
            if grad.dtype == xp.float16:
                grad = cast(grad, xp.float32).data
            F = grad.T.dot(grad) * F_scale

        return F

    def get_diagvals(self):
        gamma = self.link.gamma
        beta = self.link.beta
        if gamma is None and beta is None:
            raise RuntimeError(
                'gamma and beta must not be None as the same time')

        xp = cuda.get_array_module(gamma)
        d_gamma = self.get_diagval('gamma')
        d_beta = self.get_diagval('beta')
        if d_gamma is not None:
            d_gamma *= self.stab_coeff
        if d_beta is not None:
            d_beta *= self.stab_coeff

        setattr(self, 'diag_val_backward', d_gamma)

        if d_beta is None:
            return [d_gamma * xp.ones(gamma.size)]
        elif d_gamma is None:
            return [d_beta * xp.ones(beta.size)]
        else:
            diagvals_gamma = d_gamma * xp.ones(gamma.size)
            diagvals_beta = d_beta * xp.ones(beta.size)
            return [xp.hstack((diagvals_gamma, diagvals_beta))]

    def update_kfgrads(self):
        self.check_attr('invs')
        gamma = self.link.gamma
        beta = self.link.beta
        invs = self.invs
        kfgrads = self.compute_kfgrads(gamma, beta, invs)
        if _check_nan(kfgrads, gamma, beta):
            return
        if beta is None:
            gamma.kfgrad = kfgrads
        elif gamma is None:
            beta.kfgrad = kfgrads
        else:
            gamma.kfgrad = kfgrads[:len(gamma.grad)]
            beta.kfgrad = kfgrads[len(gamma.grad):]

    def compute_kfgrads(self, gamma, beta, invs):
        xp = cuda.get_array_module(gamma.data)
        F_inv = invs[0]
        if beta is None:
            grad = gamma.grad
        elif gamma is None:
            grad = beta.grad
        else:
            grad = xp.hstack((gamma.grad, beta.grad))

        out_dtype = grad.dtype
        if F_inv.dtype != grad.dtype:
            grad = cast(grad, F_inv.dtype).data

        kfgrad = xp.dot(F_inv, grad)

        return kfgrad.astype(out_dtype)

    def extract_for_reduce_scatter_v(self):
        arrays = self.extract_attr_from_params('grad')
        triangular = True
        arrays.append((self.cov_backward, triangular))
        return arrays
