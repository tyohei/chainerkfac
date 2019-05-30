#!/usr/bin/env python
import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainermn

import chainerkfac

import models.resnet50
import models.vgg


def main():
    parser = argparse.ArgumentParser(description='Chainer K-FAC example: CIFAR')  # NOQA
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--snapshot_interval', type=int, default=-1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume', default='')
    parser.add_argument('--optimizer', default='kfac')
    parser.add_argument('--arch', choices=['resnet50', 'vgg'], default='resnet50')  # NOQA
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    # Prepare communicator
    if not args.distributed:
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
    if args.arch == 'resnet50':
        model = L.Classifier(models.resnet50.ResNet50())
    else:
        model = L.Classifier(models.vgg.VGG())

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
        optimizer = chainer.optimizers.MomentumSGD()
        if comm is not None:
            optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    # Load the CIFAR dataset
    if rank == 0:
        train, test = chainer.datasets.get_cifar10()
    else:
        train, test = None, None
    if comm is not None:
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        test = chainermn.scatter_dataset(test, comm, shuffle=True)

    train_iter = chainer.iterators.MultithreadIterator(
        train, args.batch_size, n_threads=8)
    test_iter = chainer.iterators.MultithreadIterator(
        test, args.batch_size, n_threads=8, repeat=False, shuffle=False)

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

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

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
