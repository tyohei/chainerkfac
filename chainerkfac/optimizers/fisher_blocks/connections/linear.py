from chainer.backends import cuda
from chainer.functions import cast

from chainer.functions.connection.linear import LinearFunction
from chainerkfac.optimizers.fisher_blocks import FisherBlockConnection


class FisherBlockLinear(FisherBlockConnection):

    @property
    def funcclass(self):
        return LinearFunction

    def compute_A(self, in_data):
        x = in_data[0]
        xp = cuda.get_array_module(x)
        n, _ = x.shape
        if self.link.b is not None:
            ones = xp.ones(n, dtype=x.dtype)
            x = xp.column_stack((x, ones))

        if x.dtype == xp.float16:
            x = cast(x, xp.float32).data

        A = (x * x).mean(axis=0) if self.diagonalize else x.T.dot(x) * (1 / n)

        return A

    def compute_G(self, in_data, out_grad_data):
        gy = out_grad_data[0]
        xp = cuda.get_array_module(gy)
        n, _ = gy.shape

        gy_scale = n
        if self._loss_scale is not None:
            gy_scale *= 1.0 / self._loss_scale

        if gy.dtype == xp.float16:
            gy = gy_scale * cast(gy, xp.float32).data
        else:
            gy = gy_scale * gy

        G = xp.diag((gy * gy).mean(axis=0)) \
            if self.diagonalize else gy.T.dot(gy) * (1 / n)

        return G

    def compute_kfgrads(self, W, b, invs):
        xp = cuda.get_array_module(W.data)
        A_inv, G_inv = invs
        grad = W.grad
        if b is not None:
            grad = xp.column_stack([grad, b.grad])

        out_dtype = grad.dtype
        if A_inv.dtype != grad.dtype:
            grad = cast(grad, A_inv.dtype).data

        kfgrad = xp.dot(xp.dot(G_inv, grad), A_inv)

        return kfgrad.astype(out_dtype)
