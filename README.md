# chainerkfac

A Chainer extension for training deep neural networks with [Kronecker-Factored Approximate Curvature (K-FAC)](https://arxiv.org/abs/1503.05671).

Implementation for 
```
Kazuki Osawa, Yohei Tsuji, Yuichiro Ueno, Akira Naruse, Rio Yokota, and Satoshi Matsuoka. 
Large-Scale Distributed Second-Order Optimization Using Kronecker-Factored Approximate Curvature for Deep Convolutional Neural Networks. 
CVPR, 2019.
```
[[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Osawa_Large-Scale_Distributed_Second-Order_Optimization_Using_Kronecker-Factored_Approximate_Curvature_for_Deep_CVPR_2019_paper.pdf)]
[[poster](https://kazukiosawa.github.io/cvpr19_poster.pdf)]

## Installation
- Supported Python Versions: Python 3.6+

Clone the code from GitHub.
```
$ git clone https://github.com/tyohei/chainerkfac.git chainerkfac
```
Change the directory and install.
```
$ cd chainerkfac
$ python setup.py install
```

This table describes the additional required libraries to install before the installation of chainerkfac.

| Running environment | Additional required libraries |
|:--------------------|:------------------------------|
| Single GPU          | CuPy                          |
| Multiple GPUs       | CuPy with NCCL, MPI4py        |
| Multiple GPUs for ImageNet script | CuPy with NCCL, MPI4py, Pillow |

See [CuPy installation guide](https://docs-cupy.chainer.org/en/stable/install.html) and [ChainerMN installation guide](https://docs.chainer.org/en/stable/chainermn/installation/guide.html#chainermn-installation) for details.

## Examples

### MNIST ([codes](https://github.com/tyohei/chainerkfac/tree/master/examples/mnist)) / CIFAR-10 ([codes](https://github.com/tyohei/chainerkfac/tree/master/examples/cifar10))
Training with a single CPU
```
$ python train.py --no_cuda
```
Training with a single GPU
```
$ python train.py
```
Training with multiple GPUs (4GPUs)
```
$ mpirun -np 4 python train.py --distributed
```
### ImageNet ([codes](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet))

Training with multiple GPUs (4GPUs)
```
$ mpirun -np 4 python train.py \
<path/to/train.txt> <path/to/val.txt> \
--train_root <path/to/train_root> \
--val_root  <path/to/val_root> \
--mean ./mean.npy \
--config <path/to/config_file>
```
## Training ResNet-50 on ImageNet with large mini-batch
- Our new results with *ResNet-50-D* architecture from [Bag of Tricks for Image Classification with Convolutional Neural Networks, Tong He+, 2018](https://arxiv.org/abs/1812.01187).
- (pure *ResNet-50* was used in [our paper](https://arxiv.org/abs/1811.12019).)
- Top-1 validation accuracy for ImageNet (1,000 class) classification:

| Mini-batch size | config file | Epochs | Iterations | Top-1 Accuracy |
|:-:|:-:|:-:|:-:|:-:|
|4,096|[configs/bs4k.resnet50.128gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs4k.resnet50.128gpu.json) | 35 | 10,948 | 75.9 % |
|8,192|[configs/bs8k.resnet50.256gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs8k.resnet50.256gpu.json) | 35 | 5,478 | 76.4 % |
|16,384|[configs/bs16k.resnet50.512gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs16k.resnet50.512gpu.json) | 35 | 2,737 | 76.6 % |
|32,768|[configs/bs32k.resnet50.1024gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs32k.resnet50.1024gpu.json) | 45 | 1,760 | 76.9 % |
|65,536|[configs/bs64k.resnet50.2048gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs64k.resnet50.2048gpu.json) | 60 | 1,173 | 76.3 % |
|131,072|[configs/bs128k.resnet50.4096gpu.json](https://github.com/tyohei/chainerkfac/blob/master/examples/imagenet/configs/bs128k.resnet50.4096gpu.json) | 80 | 782 | 75.0 % |

- NOTE:
  - We recommend to use `32` for `--batchsize` (32 samples per GPU). 
  - You need to run with `N` GPUs when you use `*{N}gpu.json` config file. 
  - You need to set `--acc_iters` when you want to run training with less number of GPUs as below:
    - Mini-batch size = {samples per GPU} x {# GPUs} x {acc_iters}
    - ex) 4096 = 32 x 8 x 16
    - ex) 131072 = 32 x 8 x 512 
  - Gradients of loss and Fisher information matrix (Kronecker factors) are accumulated for `--acc_iters` iterations to build *pseudo* mini-batch size.
  - See [config files](https://github.com/tyohei/chainerkfac/tree/master/examples/imagenet/configs).


## Authors
Yohei Tsuji ([@tyohei](https://github.com/tyohei)), Kazuki Osawa ([@kazukiosawa](https://github.com/kazukiosawa)), Yuichiro Ueno ([@y1r](https://github.com/y1r)) and Akira Naruse ([@anaruse](https://github.com/anaruse))
