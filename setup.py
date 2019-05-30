#!/usr/bin/env python
from setuptools import setup


requirements = {
    'install': [
        'setuptools',
        'filelock',
        'chainer',
    ],
}

setup_requires = []
install_requires = requirements['install']

setup_kwargs = dict(
    name='chainerkfac',
    version='0.1',
    description='KFAC implementation on Chainer',
    author='Yohei Tsuji',
    author_email='t.t.t.yohei@gmail.com',
    packages=['chainerkfac',
              'chainerkfac/communicators',
              'chainerkfac/optimizers',
              'chainerkfac/optimizers/fisher_blocks',
              'chainerkfac/optimizers/fisher_blocks/connections',
              'chainerkfac/optimizers/fisher_blocks/normalizations'],
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)

setup(**setup_kwargs)
