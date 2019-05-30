import chainer
from chainer.backends import cuda
import chainer.links as L
from chainer import variable

import numpy as np
import cupy

from chainerkfac.optimizers import fisher_blocks as FB

_default_hyperparam = chainer.optimizer.Hyperparameter()
_default_hyperparam.lr = 0.001
_default_hyperparam.lr_pre = 0.001
_default_hyperparam.momentum = 0.9
_default_hyperparam.cov_ema_decay = 0.99
_default_hyperparam.damping = 0.03


def fblock_constructor(sub_linkname, sub_link, **kwargs):
    args = (sub_linkname, sub_link)
    if isinstance(sub_link, L.Linear):
        return FB.FisherBlockLinear(*args, **kwargs)
    elif isinstance(sub_link, L.Convolution2D):
        return FB.FisherBlockConvolution2D(*args, **kwargs)
    elif isinstance(sub_link, L.BatchNormalization):
        if sub_link.gamma is None and sub_link.beta is None:
            return None
        return FB.FisherBlockBatchNormalization(*args, **kwargs)
    else:
        return None


class KFACHook(chainer.function_hook.FunctionHook):

    def __init__(self, fblocks):
        self.fblocks = fblocks

    def forward_postprocess(self, func, in_data):
        for fblock in self.fblocks:
            if fblock.is_mine(func, in_data):
                fblock.forward_postprocess(func, in_data)
                return

    def backward_preprocess(self, func, in_data, out_grad_data):
        for fblock in self.fblocks:
            if fblock.is_mine(func, in_data, out_grad_data):
                fblock.backward_preprocess(func, in_data, out_grad_data)
                return


class KFACUpdateRule(chainer.optimizer.UpdateRule):

    def __init__(self, parent_hyperparam=None, adjust_momentum=False):
        super(KFACUpdateRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        self.adjust_momentum = adjust_momentum

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)

    def update(self, param):
        if self._use_fp32_update and param.dtype == np.float16:
            if self._fp32_param is None:
                self._fp32_param = variable.Variable(
                    param.array.astype(np.float32),
                    name=param.name)
            fp32_param = self._fp32_param
            if hasattr(param, 'kfgrad'):
                fp32_param.kfgrad = param.kfgrad.astype(np.float32)
            if param._loss_scale is not None:
                if hasattr(param, 'kfgrad'):
                    fp32_param.kfgrad /= param._loss_scale
        else:
            if param._loss_scale is not None:
                if hasattr(param, 'kfgrad'):
                    param.kfgrad /= param._loss_scale

        super(KFACUpdateRule, self).update(param)

        if self._use_fp32_update and param.dtype == np.float16:
            fp32_param = self._fp32_param
            if hasattr(param, 'kfgrad'):
                fp32_param.kfgrad = None

    def update_core_cpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        lr = self.hyperparam.lr
        lr_pre = self.hyperparam.lr_pre
        momentum = self.hyperparam.momentum
        if self.adjust_momentum and lr < lr_pre:
            momentum *= lr / lr_pre
        v = self.state['v']
        v *= momentum
        v -= lr * grad
        param.data += v

    def update_core_gpu(self, param):
        grad = param.kfgrad if hasattr(param, 'kfgrad') else param.grad
        if grad is None:
            return
        lr = self.hyperparam.lr
        lr_pre = self.hyperparam.lr_pre
        momentum = self.hyperparam.momentum
        if self.adjust_momentum and lr < lr_pre:
            momentum *= lr / lr_pre
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
            param += v;''',
            'kfac')(grad, lr, momentum,
                    param.data, self.state['v'])


class KFAC(chainer.optimizer.GradientMethod):

    """K-FAC optimizer.

    Args:
        lr (float): Learning rate.
        momentum (float): Momentum.
        cov_ema_decay (float): Decay rate for the exponential moving average of
            the Kronecker-factors.
        damping (float): Damping value added before taking the inverse of the
            Kronecker-factors.
        stab_coeff (float): Coefficient which is multiplied to the diagonal
            values for the BatchNormalization layer.
        constructor (float): Callable that returns a Fisher-block given a
            Chainer Link object.
        pi_type (float): Type of the norm used to compute pi.
        use_cholesky (bool): Use Cholesky decomposition to compute the inverse of
            the Kronecker-factors.
        loss_scale (float): Scaling value to avoid over/underflow in
            low-precision communication.
        adjust_momentum (bool): Adjust momentum to match the scaling with the
            learning rate.
        acc_iters (int): Accumulation iterations.
    """

    def __init__(
            self,
            lr=_default_hyperparam.lr,
            momentum=_default_hyperparam.momentum,
            cov_ema_decay=_default_hyperparam.cov_ema_decay,
            damping=_default_hyperparam.damping,
            stab_coeff=16.0,
            constructor=fblock_constructor,
            pi_type='tracenorm',
            use_cholesky=True,
            loss_scale=None,
            adjust_momentum=False,
            acc_iters=1,
    ):
        super(KFAC, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.lr_pre = lr
        self.hyperparam.momentum = momentum
        self.hyperparam.cov_ema_decay = cov_ema_decay
        self.hyperparam.damping = damping
        self.pi_type = pi_type
        self.constructor = constructor
        self.adjust_momentum = adjust_momentum
        self.stab_coeff = stab_coeff
        self.fblocks = []
        self.kfhook = None
        self._loss_scale = loss_scale
        self.constructor_kwargs = {
            'cov_ema_decay': self.hyperparam.cov_ema_decay,
            'damping': self.hyperparam.damping,
            'use_cholesky': use_cholesky,
            'pi_type': self.pi_type,
            'stab_coeff': self.stab_coeff,
            'loss_scale': self._loss_scale,
        }

        assert acc_iters >= 1, 'Accumulation iterations should be 1 or larger'
        self.acc_iters = acc_iters
        self.acc_count = 0
        self.acc_aware_iterations = 0
        self.only_pic_update_weight = False  # True only if in DistributedKFAC

    lr = chainer.optimizer.HyperparameterProxy('lr')
    lr_pre = chainer.optimizer.HyperparameterProxy('lr_pre')
    momentum = chainer.optimizer.HyperparameterProxy('momentum')
    cov_ema_decay = chainer.optimizer.HyperparameterProxy('cov_ema_decay')
    damping = chainer.optimizer.HyperparameterProxy('damping')

    def setup(self, link):
        super(KFAC, self).setup(link)

        for sub_linkname, sub_link in sorted(link.namedlinks()):
            fblock = self.constructor(sub_linkname, sub_link,
                                      **self.constructor_kwargs)
            if fblock is None:
                continue

            self.fblocks.append(fblock)
        self.kfhook = KFACHook(self.fblocks)
        return self

    def reallocate_cleared_grads(self):
        """Reallocate K-FAC gradients.

        Reallocate K-FAC gradients cleared by
        :meth:`~chainer.Variable.cleargrad`. This method allocates arrays for
        all K-FAC gradients which have :obj:`None`.
        """
        for name, param in self.target.namedparams(include_uninit=False):
            if hasattr(param, 'kfgrad') and param.kfgrad is None:
                xp = cuda.get_array_module(param.data)
                param.kfgrad = xp.zeros_like(param.data)
        super(KFAC, self).reallocate_cleared_grads()

    def create_update_rule(self):
        return KFACUpdateRule(self.hyperparam,
                              adjust_momentum=self.adjust_momentum)

    def update(self, lossfun=None, *args, **kwargs):
        if lossfun is not None:
            # Forward and backward propagations
            kfhook = self.kfhook
            with kfhook:
                use_cleargrads = getattr(self, '_use_cleargrads', True)
                loss = lossfun(*args, **kwargs)
                if use_cleargrads:
                    self.target.cleargrads()
                else:
                    self.target.zerograds()
                loss.backward(loss_scale=self._loss_scale)
            del loss

        if self.acc_iters > 1:
            _continue_accumulating = self.accumulate()
            if _continue_accumulating:
                return

        self.reallocate_cleared_grads()

        self.call_hooks('pre')

        self.t += 1
        self.update_kfgrads(self.fblocks)

        if not self.only_pic_update_weight:
            for param in self.target.params():
                param.update()

        self.reallocate_cleared_grads()

        self.call_hooks('post')

        self.acc_aware_iterations += 1
        chainer.report({'acc_aware_iterations': self.acc_aware_iterations})

    def accumulate(self):
        if self.acc_count == 0:
            for param in self.target.params():
                param.acc_grad = None

            for fblock in self.fblocks:
                fblock.acc_covs = None

        self.accumulate_grads()
        self.accumulate_covs()
        self.acc_count += 1

        if self.acc_count == self.acc_iters:
            self.acc_count = 0
            for param in self.target.params():
                param.grad = param.acc_grad

            for fblock in self.fblocks:
                fblock.covs = fblock.acc_covs
            return False  # not continue
        elif self.acc_count < self.acc_iters:
            return True  # continue
        else:
            raise ValueError(
                'acc_count {} should not be larger than acc_iters {}'.format(
                    self.acc_count, self.acc_iters))

    def accumulate_grads(self):
        """Accumulate gradient for each param"""
        for param in self.target.params():
            new_grad = param.grad * (1 / self.acc_iters)
            if not hasattr(param, 'acc_grad') or param.acc_grad is None:
                param.acc_grad = new_grad
            else:
                param.acc_grad += new_grad

    def accumulate_covs(self):
        """Accumulate covariance for each layer"""
        for fblock in self.fblocks:
            fblock.check_attr('covs')  # Check covs is not None
            new_covs = [cov * (1 / self.acc_iters) for cov in fblock.covs]
            if not hasattr(fblock, 'acc_covs') or fblock.acc_covs is None:
                fblock.acc_covs = new_covs
            else:
                for acc_cov, new_cov in zip(fblock.acc_covs, new_covs):
                    acc_cov += new_cov

    def update_kfgrads(self, fblocks):
        for fblock in fblocks:
            fblock.update_cov_emas()
            fblock.update_invs()
            fblock.update_kfgrads()


class DistributedKFAC(KFAC):

    """K-FAC optimizer for distributed execution.
    
    Args:
        ...
        comm (~chainerkfac.communicators.KFACCommunicator): Communicator.
        communicate_after_forward (bool): Call All-Reduce after forward
            (before backprop).
        only_pic_update_weight (bool): Only update the parameters that is
            in charge.
    """

    def __init__(self, comm, *args,
                 communicate_after_forward=False,
                 only_pic_update_weight=True,
                 **kwargs):
        super(DistributedKFAC, self).__init__(*args, **kwargs)
        self.comm = comm
        self.only_pic_update_weight = only_pic_update_weight
        self.stream = cupy.cuda.Stream(non_blocking=True) \
            if cuda.available and communicate_after_forward else None
        self.cuda_event = cupy.cuda.stream.Event(block=True,
                                                 disable_timing=True) \
            if cuda.available and communicate_after_forward else None
        self.target_params = []
        self.constructor_kwargs['communicate_after_forward'] = communicate_after_forward  # NOQA
        self.communicate_after_forward = communicate_after_forward

    def setup(self, link):
        super(DistributedKFAC, self).setup(link)
        local_size = self.comm.size
        local_rank = self.comm.rank
        indices = np.array_split(np.arange(len(self.fblocks)), local_size)
        indices = [local_indices.tolist() for local_indices in indices]
        local_indices = indices[local_rank]
        local_fblocks = [self.fblocks[i] for i in local_indices]

        self.indices = indices
        self.local_indices = local_indices
        self.local_fblocks = local_fblocks
        setattr(self.comm, 'indices', indices)

    def update(self, lossfun=None, *args, **kwargs):
        if lossfun is not None:
            # Forward and backward propagations
            kfhook = self.kfhook
            with kfhook:
                use_cleargrads = getattr(self, '_use_cleargrads', True)
                loss = lossfun(*args, **kwargs)

                # Reduce-Scatter-V
                self.reduce_scatter_v_after_forward()

                if use_cleargrads:
                    self.target.cleargrads()
                else:
                    self.target.zerograds()

                loss.backward(loss_scale=self._loss_scale)

            del loss

        if self.is_changed(self.target):
            self.comm.bcast_data(self.target)
        else:
            # Reduce+ScatterV
            self.reduce_scatter_v_after_backward()

            super(DistributedKFAC, self).update()

    def reduce_scatter_v_after_forward(self):
        if not self.communicate_after_forward:
            return

        if self.cuda_event is not None:
            self.cuda_event.record(cupy.cuda.Stream.null)
            self.stream.wait_event(self.cuda_event)

        with self.stream:
            arrays = extract(
                self.fblocks, self.indices,
                lambda block:
                block.extract_for_reduce_scatter_v_after_forward())

            self.comm.reduce_scatter_v_arrays(arrays, self.stream)

    def reduce_scatter_v_after_backward(self):
        if self.cuda_event is not None:
            # Wait the overlapping stream to finish their procedures
            self.cuda_event.record(self.stream)
            cuda.cupy.cuda.Stream.null.wait_event(self.cuda_event)

        arrays = extract(self.fblocks, self.indices,
                         lambda block: block.extract_for_reduce_scatter_v())

        self.comm.reduce_scatter_v_arrays(arrays)

    def is_changed(self, target):
        previous_params = self.target_params
        self.target_params = [(name, param.data is not None)
                              for name, param in sorted(target.namedparams())]
        if len(previous_params) != len(self.target_params):
            return True

        for param1, param2 in zip(self.target_params, previous_params):
            if (param1[0] != param2[0]) or param1[1] != param2[1]:
                return True
        return False

    def allocate_asgrad(self, attr):
        for block in self.fblocks:
            for _, param in sorted(block.link.namedparams()):
                if not hasattr(param, attr):
                    # We need to allocate memory space for receiving data
                    _grad = param.grad.copy()
                    _grad.fill(0.)
                    setattr(param, attr, _grad)

    def update_kfgrads(self, fblocks):
        super(DistributedKFAC, self).update_kfgrads(self.local_fblocks)

        # AllGatherV
        self.allocate_asgrad('kfgrad')

        if self.only_pic_update_weight:
            for fblock in self.local_fblocks:
                for param in fblock.link.params():
                    param.update()
            # AllGatherV weights
            arrays = extract(fblocks, self.indices,
                             lambda block:
                             block.extract_for_all_gather_v('data'))

        else:
            # AllGatherV kfgradients
            arrays = extract(fblocks, self.indices,
                             lambda block:
                             block.extract_for_all_gather_v('kfgrad'))

        self.comm.all_gather_v_arrays(arrays)


def extract(blocks, indices, extractor):
    arrays = []
    for local_indices in indices:
        local_arrays = []
        for index in local_indices:
            for array in extractor(blocks[index]):
                local_arrays.append(array)
        arrays.append(local_arrays)
    return arrays
