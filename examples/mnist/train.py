#!/usr/bin/env python
import argparse
import json
import numpy
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainermn

import chainerkfac


# Network definition
class MLP(chainer.Chain):

    def __init__(self):
        n_hid = 1000
        n_out = 10
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hid)
            self.l2 = L.Linear(None, n_hid)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# Convolutional Neural Network
class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, 5)
            self.conv2 = L.Convolution2D(20, 50, 5)
            self.l3 = L.Linear(None, 500)
            self.l4 = L.Linear(None, 10)

    def forward(self, x):
        x = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        x = F.max_pooling_2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='Chainer K-FAC example: MNIST')  # NOQA
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--snapshot_interval', type=int, default=-1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', default='')
    parser.add_argument('--optimizer', default='kfac')
    parser.add_argument('--arch', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    # Prepare communicator
    if not args.distributed:
        # Single process execution
        comm = None
        rank = 0
        device = -1 if args.no_cuda else 0
    else:
        # Multiple processes execution, constructs a communicator.
        # chainerkfac uses different method to create a communicator from
        # chainermn.
        if args.optimizer == 'kfac':
            comm = chainerkfac.create_communicator('pure_nccl')
        else:
            comm = chainermn.create_communicator('pure_nccl')
        rank = comm.rank
        device = comm.intra_rank
        if rank == 0:
            print('======== DISTRIBUTED TRAINING ========')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.arch == 'mlp':
        model = L.Classifier(MLP())
        in_ndim = 1  # input dimentions
    else:
        model = L.Classifier(CNN())
        in_ndim = 3  # input dimentions

    if device >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    if args.optimizer == 'kfac':
        if comm is None:
            optimizer = chainerkfac.optimizers.KFAC()
        else:
            optimizer = chainerkfac.optimizers.DistributedKFAC(comm)
    else:
        optimizer = chainer.optimizers.Adam()
        if comm is not None:
            optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    # Load the MNIST dataset
    if rank == 0:
        train, test = chainer.datasets.get_mnist(ndim=in_ndim)
    else:
        train, test = None, None
    if comm is not None:
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        test = chainermn.scatter_dataset(test, comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.num_epochs, 'epoch'),
                               out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    if comm is not None:
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if rank == 0:
        # Take a snapshot for each specified epoch
        snapshot_interval = args.num_epochs \
            if args.snapshot_interval == -1 else max(1, args.snapshot_interval)
        trainer.extend(extensions.snapshot(),
                       trigger=(snapshot_interval, 'epoch'))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())

        # Save two plot images to the result dir
        if args.plot and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))

        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again,
        # and "validation" refers to the default name of the Evaluator
        # extension. Entries other than 'epoch' are reported by the Classifier
        # link, called by either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
