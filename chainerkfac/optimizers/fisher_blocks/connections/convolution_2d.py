from chainer.backends import cuda

import cupy
import numpy as np

from chainer.functions import cast
from chainer.functions import im2col

from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainerkfac.optimizers.fisher_blocks import FisherBlockConnection


class FisherBlockConvolution2D(FisherBlockConnection):

    @property
    def funcclass(self):
        return Convolution2DFunction

    def compute_A(self, in_data):
        x = in_data[0]
        ksize, stride, pad = \
            self._link.ksize, self._link.stride[0], self._link.pad[0]
        xp = cuda.get_array_module(x)

        x = im2col(x, ksize, stride, pad).data
        x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        n, ho, wo, _ = x.shape
        x = x.reshape(n * ho * wo, -1)
        if self._link.b is not None:
            ones = xp.ones(x.shape[0], dtype=x.dtype)
            x = xp.column_stack((x, ones))

        A_scale = 1 / n
        if x.dtype == xp.float16:
            x = cast(x, xp.float32).data
            A = x.T.dot(x) * A_scale
        else:
            A = x.T.dot(x) * A_scale

        return A

    def compute_G(self, in_data, out_grad_data):
        gy = out_grad_data[0]
        xp = cuda.get_array_module(gy)

        gy = gy.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        n, ho, wo, _ = gy.shape

        gy = gy.reshape(n * ho * wo, -1)

        gy_scale = n
        if self._loss_scale is not None:
            gy_scale *= 1.0 / self._loss_scale

        if self.diagonalize:
            if gy.dtype == xp.float16:
                gy = gy_scale * cast(gy, xp.float32).data
            else:
                gy = gy_scale * gy
            G = xp.diag((gy * gy).mean(axis=0))
        else:
            G_scale = 1 / (n * ho * wo) * (gy_scale ** 2)
            if gy.dtype == xp.float16:
                gy = cast(gy, xp.float32).data
            G = gy.T.dot(gy) * G_scale

            diag = getattr(self.link, 'diag', False)
            if diag:
                G = xp.diag(xp.diag(G))

        return G

    def compute_kfgrads(self, W, b, invs):
        xp = cuda.get_array_module(W.data)
        A_inv, G_inv = invs

        grad = W.grad
        grad = grad.reshape(grad.shape[0], -1)

        if b is not None:
            grad = xp.column_stack([grad, b.grad])

        out_dtype = grad.dtype
        if A_inv.dtype != grad.dtype:
            grad = cast(grad, A_inv.dtype).data

        kfgrad = xp.dot(xp.dot(G_inv, grad), A_inv)

        return kfgrad.astype(out_dtype)

