from chainer.backends import cuda

import cupy
import numpy as np

from chainerkfac.optimizers.cholesky_inverse import inverse

import warnings


PI_TYPE_TRACENORM = 'tracenorm'


def get_diagval(link, attrname, damping):
    param = getattr(link, attrname, None)
    if param is None:
        return
    r = getattr(param, 'l2_coef', 0.0)
    r = max(r, damping)
    return r


def compute_pi_tracenorm(lcov, rcov):
    """Compute pi using tracenorm

    Computes the scalar constant pi for Tikhonov regularization/damping.
    $\pi = \sqrt{ (trace(A) / dim(A)) / (trace(B) / dim(B)) }$
    See section 6.3 of https://arxiv.org/pdf/1503.05671.pdf for details.
    """  # NOQA

    def compute_trace(cov):
        if cov.ndim == 1:
            return cov.sum()
        else:
            return cov.trace()

    xp = cuda.get_array_module([lcov, rcov])
    with cuda.get_device_from_array([lcov, rcov]):
        lnorm = compute_trace(lcov) * rcov.shape[0]
        rnorm = compute_trace(rcov) * lcov.shape[0]
        pi = xp.sqrt(lnorm / rnorm, dtype=xp.float32)
    return pi


def compute_pi(lcov, rcov, pi_type=PI_TYPE_TRACENORM):
    if pi_type == PI_TYPE_TRACENORM:
        return compute_pi_tracenorm(lcov, rcov)
    else:
        return 1.0


class FisherBlock(object):

    """Base class for Fisher-block.

    Args:
        linkname (string): Name to identify this FisherBlock object (do not
            have to be identical, only for debug use).
        link (~chainer.Link): Chainer Link object corresponding to this
            FisherBlock object.
        cov_ema_decay (float): Decay rate for the exponential moving average of
            the Kronecker-factors.
        damping (float): Damping value added before taking the inverse of the
            Kronecker-factors.
        communicate_after_forward (bool): Call All-Reduce after forward
            (before backprop).
        pi_type (float): Type of the norm used to compute pi.
        use_cholesky (bool): Use Cholesky decomposition to compute the inverse of
            the Kronecker-factors.
        stab_coeff (float): Coefficient which is multiplied to the diagonal
            values for the BatchNormalization layer.
        loss_scale (float): Scaling value to avoid over/underflow in
            low-precision communication.
    """

    def __init__(
            self,
            linkname,
            link,
            cov_ema_decay,
            damping,
            communicate_after_forward=False,
            pi_type=PI_TYPE_TRACENORM,
            use_cholesky=True,
            stab_coeff=64.0,
            loss_scale=None,
    ):
        self._link = link
        self._linkname = linkname
        self._linkclass = link.__class__
        self._communicate_after_forward = communicate_after_forward
        self._pi_type = pi_type
        self._use_cholesky = use_cholesky
        self._loss_scale = loss_scale
        self.cov_ema_decay = cov_ema_decay
        self.damping = damping
        self.stab_coeff = stab_coeff
        self.xp = link.xp
        self.covs = None
        self.invs = None
        self.cov_emas = None
        self.diagonalize = False

    @property
    def linkname(self):
        # Used for distributed K-FAC
        return self._linkname

    @property
    def link(self):
        # Used for distributed K-FAC
        return self._link

    @property
    def funcclass(self):
        raise NotImplementedError

    @property
    def cov_forward(self):
        raise NotImplementedError

    @property
    def cov_backward(self):
        raise NotImplementedError

    @property
    def inv_forward(self):
        raise NotImplementedError

    @property
    def inv_backward(self):
        raise NotImplementedError

    def debug(self):
        return getattr(self._link, 'debug', False)

    def is_mine(self, func, in_data, out_grad_data=None):
        raise NotImplementedError

    def forward_postprocess(self, func, in_data):
        raise NotImplementedError

    def backward_preprocess(self, func, in_data, out_grad_data):
        raise NotImplementedError

    def check_attr(self, name):
        if getattr(self, name) is None:
            raise RuntimeError(
                '{} is None:\nlinkname: {}\nlinkclass: {}'.format(
                    name, self._linkname, self._linkclass))

    def get_diagval(self, attr):
        return get_diagval(self._link, attr, self.damping)

    def update_cov_emas(self):
        self.check_attr('covs')
        r = self.cov_ema_decay
        if r == 1.0 or self.cov_emas is None:
            # To avoid broken in inverse (inverse is implemented using in-place)
            self.cov_emas = self.covs.copy()
        else:
            self.cov_emas = [r*cov + (1 - r)*cov_ema for cov, cov_ema
                             in zip(self.covs, self.cov_emas)]

    def update_invs(self):
        self.check_attr('cov_emas')
        xp = self.xp
        self.invs = []
        for cov_ema, diagvals in zip(self.cov_emas, self.get_diagvals()):
            if cov_ema.ndim == 1:
                inv = 1 / (cov_ema + diagvals)
            else:
                _cov_ema = cov_ema.copy()

                xp.fill_diagonal(cov_ema, xp.diagonal(cov_ema) + diagvals)

                inv = inverse(cov_ema, self._use_cholesky)

            self.invs.append(inv)

    def get_diagvals(self):
        raise NotImplementedError

    def update_kfgrads(self):
        raise NotImplementedError

    def extract_attr_from_params(self, attrname, triangular=False):
        """Extracts arrays from all ``Parameter``s
        """
        arrays = []
        for _, param in sorted(self.link.namedparams()):
            x = getattr(param, attrname, None)
            if x is not None:
                arrays.append((x, triangular))
        return arrays

    def extract_for_reduce_scatter_v_after_forward(self):
        if self._communicate_after_forward:
            triangular = True
            return [(self.cov_forward, triangular)]
        else:
            return []

    def extract_for_reduce_scatter_v(self):
        arrays = self.extract_attr_from_params('grad')
        if not self._communicate_after_forward:
            triangular = True
            arrays.append((self.cov_forward, triangular))
        triangular = True
        arrays.append((self.cov_backward, triangular))
        return arrays

    def extract_for_all_gather_v(self, target='kfgrad'):
        arrays = self.extract_attr_from_params(target)
        triangular = False
        return arrays

