from chainer.backends import cuda

from chainerkfac.optimizers.fisher_block import compute_pi
from chainerkfac.optimizers.fisher_block import FisherBlock


class FisherBlockConnection(FisherBlock):

    def __init__(self, *args, **kwargs):
        self._A = None
        self._G = None
        super(FisherBlockConnection, self).__init__(*args, **kwargs)

    @property
    def cov_forward(self):
        return self.covs[0]

    @property
    def cov_backward(self):
        return self.covs[1]

    @property
    def inv_forward(self):
        return self.invs[0]

    @property
    def inv_backward(self):
        return self.invs[1]

    def is_mine(self, func, in_data, out_grad_data=None):
        if not isinstance(func, self.funcclass):
            return False
        if in_data[1] is not self.link.W.data:
            return False
        return True

    def forward_postprocess(self, func, in_data):
        self._A = self.compute_A(in_data)
        self.covs = [self._A, self._G]

    def backward_preprocess(self, func, in_data, out_grad_data):
        self._G = self.compute_G(in_data, out_grad_data)
        self.covs = [self._A, self._G]

    def compute_A(self, in_data):
        raise NotImplementedError

    def compute_G(self, in_data, out_grad_data):
        raise NotImplementedError

    def update_kfgrads(self):
        self.check_attr('invs')
        W = self.link.W
        b = self.link.b
        invs = self.invs
        kfgrads = self.compute_kfgrads(W, b, invs)
        if b is not None:
            W.kfgrad = kfgrads[:, :-1].reshape(W.shape)
            b.kfgrad = kfgrads[:, -1].reshape(b.shape)
        else:
            W.kfgrad = kfgrads.reshape(W.shape)

    def compute_kfgrads(self, W, b, invs):
        raise NotImplementedError

    def get_diagvals(self):
        A, G = self.cov_emas
        xp = cuda.get_array_module(A)
        rW = self.get_diagval('W') ** 0.5
        diagvalsA = rW * xp.ones(A.shape[0])
        diagvalsG = rW * xp.ones(G.shape[0])
        if self.link.b is not None:
            diagvalsA[-1] = rW
        pi = compute_pi(A, G, self._pi_type)
        setattr(self, 'diag_val_forward', pi * rW)
        setattr(self, 'diag_val_backward', (1 / pi) * rW)
        return [pi * diagvalsA, (1 / pi) * diagvalsG]

