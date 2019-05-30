# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
from chainer.backends import cuda
from chainer import initializers
from chainer import configuration
import chainer.functions as F
import chainer.links as L

from softmax_cross_entropy import softmax_cross_entropy

import math
import numpy as np
import random


class NormalM(initializers.Normal):

    def __init__(self, scale=0.05, mean=1.0, dtype=None):
        self.mean = mean
        super(NormalM, self).__init__(scale, dtype)

    def __call__(self, array):
        super(NormalM, self).__call__(array)
        array[...] += self.mean


_bn_decay = 0.9
_bn_use_param = True
_bn_initial_gamma = NormalM()
_bn_initial_beta = None
_bn_eps = 2e-5

_fc_nobias = False
_fc_use_dropout = False
_fc_dropout_rate = 0.2

_conv_nw = True
_fc_nw = True
_bn_nw = False


class InitialLinearW(initializers.Normal):

    def __init__(self, dtype=None):
        super(InitialLinearW, self).__init__(dtype)

    def __call__(self, array):
        co, ci = array.shape
        scale = np.sqrt(2.0 / ci)
        initializers.Normal(scale)(array)


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()
        self.stride = stride

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(
                ch,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv2 = L.Convolution2D(
                ch, ch, 3, stride, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(
                ch,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(
                out_size,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, 1, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(
                out_size,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            setattr(self.conv1.W, 'normalize_weight', _conv_nw)
            setattr(self.conv2.W, 'normalize_weight', _conv_nw)
            setattr(self.conv3.W, 'normalize_weight', _conv_nw)
            setattr(self.conv4.W, 'normalize_weight', _conv_nw)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        if self.stride > 1:
            h2 = F.average_pooling_2d(x, 2, stride=self.stride)
            h2 = self.bn4(self.conv4(h2))
        else:
            h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(
                ch,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(
                ch,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(
                in_size,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            setattr(self.conv1.W, 'normalize_weight', _conv_nw)
            setattr(self.conv2.W, 'normalize_weight', _conv_nw)
            setattr(self.conv3.W, 'normalize_weight', _conv_nw)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self, n=1, mixup_alpha=None, running_mixup=False,
                 re_area_rl=0.02, re_area_rh=0.2, re_aspect_rl=0.33, re_rate=0.0):
        n = int(n)
        initialW = initializers.HeNormal()
        fc_initialW = InitialLinearW()
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(
                3, 32//n, 3, 2, 1, initialW=initialW)
            self.bn1_1 = L.BatchNormalization(
                32//n,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv1_2 = L.Convolution2D(
                32, 32//n, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn1_2 = L.BatchNormalization(
                32//n,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.conv1_3 = L.Convolution2D(
                32, 64//n, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn1_3 = L.BatchNormalization(
                64//n,
                decay=_bn_decay,
                eps=_bn_eps,
                initial_gamma=_bn_initial_gamma,
                initial_beta=_bn_initial_beta,
                use_gamma=_bn_use_param,
                use_beta=_bn_use_param,
                )
            self.res2 = Block(3, 64//n, 64//n, 256//n, 1)
            self.res3 = Block(4, 256//n, 128//n, 512//n)
            self.res4 = Block(6, 512//n, 256//n, 1024//n)
            self.res5 = Block(3, 1024//n, 512//n, 2048//n)
            self.fc = L.Linear(2048//n, 1000//n, nobias=_fc_nobias,
                               initialW=fc_initialW)
        self.mixup_alpha = mixup_alpha
        self.running_mixup = running_mixup
        # re: random eracing
        self.re_area_rl = re_area_rl
        self.re_area_rh = re_area_rh
        self.re_aspect_rl = re_aspect_rl
        self.re_rate = re_rate

        self.nclass = 1000//n
        self.prior_x = None
        setattr(self.conv1_1.W, 'normalize_weight', _conv_nw)
        setattr(self.conv1_2.W, 'normalize_weight', _conv_nw)
        setattr(self.conv1_3.W, 'normalize_weight', _conv_nw)
        setattr(self.fc.W, 'normalize_weight', _fc_nw)

    def _mixup(self, x, t):
        tt = None
        if configuration.config.train and self.mixup_alpha is not None:
            xp = cuda.get_array_module(x)
            nbatch = x.shape[0]
            tt = xp.zeros((nbatch, self.nclass), dtype=x.dtype)
            tt[np.arange(nbatch), t] = 1
            alpha = self.mixup_alpha
            lam = np.random.beta(alpha, alpha, nbatch).astype(x.dtype)
            lam = np.maximum(lam, 1.0 - lam)
            if xp is cuda.cupy:
                lam = cuda.to_gpu(lam)
            if self.prior_x is None:
                mixup_x = x
                mixup_tt = None
            else:
                lam = lam.reshape(nbatch, 1, 1, 1)
                mixup_x = lam * x + (1.0 - lam) * self.prior_x
                lam = lam.reshape(nbatch, 1)
                mixup_tt = lam * tt + (1.0 - lam) * self.prior_tt
            self.prior_x = x
            self.prior_tt = tt
            x = mixup_x
            tt = mixup_tt
            if self.running_mixup:
                self.prior_x = x
                if tt is not None:
                    self.prior_tt = tt
        return x, t, tt

    def _random_erase(self, x, area_rl, area_rh, aspect_rl, rate):
        n, _, h, w = x.shape
        area_size = h * w
        for i in range(n):
            if random.uniform(0, 1) > rate:
                continue
            erase_area_size = area_size * random.uniform(area_rl, area_rh)
            aspect_ratio = random.uniform(aspect_rl, 1.0)
            erase_h = int(math.sqrt(erase_area_size * aspect_ratio) + .5)
            erase_w = int(math.sqrt(erase_area_size / aspect_ratio) + .5)
            if random.randint(0, 1):
                erase_h, erase_w = erase_w, erase_h
            erase_h = min(erase_h, h)
            erase_w = min(erase_w, w)
            top = random.randint(0, max(h - erase_h - 1, 0))
            left = random.randint(0, max(w - erase_w - 1, 0))
            bottom = top + erase_h
            right = left + erase_w
            bottom = min(bottom, h)
            right = min(right, w)
            x[i, :, top:bottom, left:right] = 0
        return x

    def __call__(self, x, t):
        if self.mixup_alpha is not None:
            x, t, tt = self._mixup(x, t)

        if configuration.config.train:
            if self.re_rate > 0:
                x = self._random_erase(x, self.re_area_rl, self.re_area_rh,
                                       self.re_aspect_rl, self.re_rate)

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        if _fc_use_dropout:
            h = F.dropout(h, _fc_dropout_rate)
        h = self.fc(h)

        if self.mixup_alpha is not None:
            loss = softmax_cross_entropy(h, t, tt)
        else:
            loss = softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

