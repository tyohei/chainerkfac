#!/usr/bin/env python

from __future__ import print_function
import argparse
import json
import math
import multiprocessing
import numpy as np
import os
import shutil
import socket
import sys
from importlib import import_module

import chainer
import chainer.cuda
from chainer.configuration import config
from chainer.backends import cuda
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import chainermn

import chainerkfac
import datasets


class LrPolynomialDecay(extension.Extension):

    def __init__(self, ref_lr, epoch_start, epoch_end, p=2, start_rate=2e-5):
        self.ref_lr = ref_lr
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.p = p
        self.start_rate = start_rate

    def lr(self, epoch):
        if epoch < self.epoch_start:  # linear warmup
            rate = 1.0
        elif epoch < self.epoch_end:
            rate = 1.0 - (epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)  # NOQA
            rate = rate ** self.p
        else:
            rate = 0.0
        return self.ref_lr * rate

    def __call__(self, trainer):
        epoch = trainer.updater.epoch_detail
        lr = self.lr(epoch)
        optimizer = trainer.updater.get_optimizer('main')
        lr_pre = getattr(optimizer, 'lr')
        setattr(optimizer, 'lr_pre', lr_pre)
        setattr(optimizer, 'lr', lr)


class NormalizeWeightUR(object):

    name = 'NormalizeWeightUR'
    timing = 'post'

    def __init__(self, skip_scale_comp=False):
        self.skip_scale_comp = skip_scale_comp
        self.warmup_len = 2
        self.max_skip = 10
        self.tolerance = 0.98
        self.eps = 1e-9

    def _normalize_weight(self, rule, param, target_norm=1.0):
        w = param.data
        xp = cuda.get_array_module(w)
        with cuda.get_device_from_array(w):

            calc_scale = False
            calc_interval = False
            pre_interval = 1
            if self.skip_scale_comp is False or rule.t < self.warmup_len:
                calc_scale = True
            else:
                if rule.t >= param._t_next:
                    calc_scale = True
                    calc_interval = True
                    pre_interval = param._t_next - param._t

            if calc_scale:
                w_norm = xp.linalg.norm(w)
                scale = target_norm / (w_norm + self.eps)

                if calc_interval:
                    right_scale = math.pow(
                        scale * param._scale ** (pre_interval - 1),
                        1 / pre_interval)
                    error_rate = param._scale / right_scale
                    if error_rate > 1:
                        error_rate = 1 / error_rate
                    max_interval = min(self.max_skip, math.log(self.tolerance, error_rate) + 1)  # NOQA
                    interval = int(min(max_interval, pre_interval + 1))
                    param._scale = right_scale
                else:
                    interval = 1
                    param._scale = scale

                param._t = rule.t
                param._t_next = rule.t + interval
            else:
                scale = param._scale

            if scale < 1:
                w[...] *= scale

    def __call__(self, rule, param):
        w = param.data
        if w is None:
            return
        target_norm = np.sqrt(2.0 * w.shape[0])
        self._normalize_weight(rule, param, target_norm)


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def get_arch(arch_file, arch_name):
    ext = os.path.splitext(arch_file)[1]
    mod_path = '.'.join(os.path.split(arch_file)).replace(ext, '')
    mod = import_module(mod_path)
    return getattr(mod, arch_name)


def main():
    # Check if GPU is available
    if not chainer.cuda.available:
        raise RuntimeError("ImageNet requires GPU support.")

    parser = argparse.ArgumentParser(
        description='Training ResNet50 on ImageNet')
    # Data
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--train_root', default='.')
    parser.add_argument('--val_root', default='.')
    parser.add_argument('--mean', default='mean.npy')
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument('--iterator', default='thread')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default='models/resnet50.py')
    parser.add_argument('--arch_name', type=str, default='ResNet50')
    parser.add_argument('--initmodel')
    parser.add_argument('--resume',  default='')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=16)
    parser.add_argument('--acc_iters', type=int, default=1)
    parser.add_argument('--epoch', '-E', type=int, default=36)
    parser.add_argument('--normalize_weight', action='store_true', default=True)  # NOQA
    parser.add_argument('--nw_skip_scale_comp', action='store_true', default=False)  # NOQA
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=8.18e-3)
    parser.add_argument('--lr_plan', default='polynomial')
    parser.add_argument('--epoch_lr_decay_start', type=float, default=1)
    parser.add_argument('--polynomial_decay_p', type=float, default=11)
    parser.add_argument('--polynomial_epoch', type=float, default=53)
    parser.add_argument('--momentum', type=float, default=0.997)
    parser.add_argument('--adjust_momentum', action='store_true', default=True)  # NOQA
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--running_mixup', action='store_true', default=True)
    parser.add_argument('--re_rate', type=float, default=0.5)
    parser.add_argument('--re_area_rl', type=float, default=0.02)
    parser.add_argument('--re_area_rh', type=float, default=0.25)
    parser.add_argument('--re_aspect_rl', type=float, default=0.3)
    parser.add_argument('--cov_ema_decay', type=float, default=1.0)
    parser.add_argument('--damping', type=float, default=2.5e-4)
    parser.add_argument('--use_tensor_core', action='store_true', default=False)  # NOQA
    parser.add_argument('--communicate_after_forward', action='store_true', default=False)  # NOQA
    # Other
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--config_out', default='config.json')
    parser.add_argument('--out', '-o', default='result')

    args = parser.parse_args()
    dict_args = vars(args)

    # ======== Load config file ========
    if args.config is not None:
        with open(args.config) as f:
            _config = json.load(f)
        dict_args.update(_config)

    # ======== Create communicator ========
    comm = chainerkfac.create_communicator('pure_nccl')
    device = comm.intra_rank

    # ======== Create model ========
    kwargs = {
        'mixup_alpha': args.mixup_alpha,
        'running_mixup': args.running_mixup,
        're_area_rl': args.re_area_rl,
        're_area_rh': args.re_area_rh,
        're_aspect_rl': args.re_aspect_rl,
        're_rate': args.re_rate,
    }
    arch = get_arch(args.arch_file, args.arch_name)
    model = arch(**kwargs)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # ======== Copy model to GPU ========
    try:
        chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
        model.to_gpu()
    except chainer.cuda.cupy.cuda.runtime.CUDARuntimeError as e:
        print('[ERROR] Host: {}, GPU ID: {}'
              .format(socket.gethostname(), device), file=sys.stderr)
        raise e

    # ======== Create dataset ========
    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(args.mean)
    if comm.rank == 0:
        train = datasets.read_pairs(args.train)
        val = datasets.read_pairs(args.val)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)
    train = datasets.CroppingImageDatasetIO(
        train, args.train_root, mean, model.insize, model.insize)
    val = datasets.CroppingImageDatasetIO(
        val, args.val_root, mean, model.insize, model.insize, False)

    # ======== Create iterator ========
    if args.iterator == 'process':
        # We need to change the start method of multiprocessing module if we
        # are using InfiniBand and MultiprocessIterator. This is because
        # processes often crash when calling fork if they are using Infiniband.
        # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
        multiprocessing.set_start_method('forkserver')
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, args.val_batchsize, n_processes=args.loaderjob, repeat=False,
            shuffle=False)
    elif args.iterator == 'thread':
        train_iter = chainer.iterators.MultithreadIterator(
            train, args.batchsize, n_threads=args.loaderjob)
        val_iter = chainer.iterators.MultithreadIterator(
            val, args.val_batchsize, n_threads=args.loaderjob, repeat=False,
            shuffle=False)
    else:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val, args.val_batchsize, repeat=False, shuffle=False)

    # ======== Create optimizer ========
    optimizer = chainerkfac.optimizers.DistributedKFAC(
        comm,
        lr=args.lr,
        momentum=args.momentum,
        cov_ema_decay=args.cov_ema_decay,
        damping=args.damping,
        acc_iters=args.acc_iters,
        adjust_momentum=args.adjust_momentum,
        communicate_after_forward=args.communicate_after_forward,
    )
    optimizer.setup(model)
    optimizer.use_fp32_update()

    if args.normalize_weight:
        link = getattr(optimizer, 'target')
        for param in link.params():
            if getattr(param, 'normalize_weight', False):
                param.update_rule.add_hook(
                    NormalizeWeightUR(skip_scale_comp=args.nw_skip_scale_comp))
                                      

    if comm.rank == 0:
        print('indices: {}'.format(optimizer.indices))

    # ======== Create updater ========
    updater = training.StandardUpdater(train_iter, optimizer, device=device)

    # ======== Create trainer ========
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    # ======== Extend trainer ========
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')
    # Create a multi node evaluator from an evaluator.
    evaluator = TestModeEvaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)
    # Reduce the learning rate
    if args.lr_plan == 'polynomial':
        epoch_end = max(args.epoch, args.polynomial_epoch)
        trainer.extend(
            LrPolynomialDecay(args.lr, args.epoch_lr_decay_start, epoch_end,
                              p=args.polynomial_decay_p),
            trigger=(args.acc_iters, 'iteration'))

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'elapsed_time', 'main/loss',
            'validation/main/loss', 'main/accuracy',
            'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        if args.stats:
            trainer.extend(extensions.ParameterStatistics(model))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if comm.rank == 0:
        hyperparams = optimizer.hyperparam.get_dict()
        for k, v in hyperparams.items():
            print('{}: {}'.format(k, v))

    # ======== Save configration ========
    os.makedirs(args.out, exist_ok=True)
    my_config = {}
    my_config['args'] = vars(args)
    my_config['hyperparams'] = optimizer.hyperparam.get_dict()

    with open(os.path.join(args.out, args.config_out), 'w') as f:
        r = json.dumps(my_config)
        f.write(r)

    # Copy this file to args.out
    shutil.copy(os.path.realpath(__file__), args.out)

    chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    config.autotune = True
    config.cudnn_fast_batch_normalization = True

    trainer.run()


if __name__ == '__main__':
    main()
