#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import chainer.optimizers as O
import chainer.serializers as S

import fcn
from fcn.models import FCNbleaney
from fcn.models import VGG16
from fcn import bleaney



def main(gpu):

    # setup dataset
    dataset = bleaney.SegmentationClassDataset()
    n_class = len(dataset.target_names)

    # setup model
    model = FCNbleaney(n_class=n_class)
    if gpu != -1:
        model.to_gpu(gpu)

    # setup optimizer
    optimizer = O.MomentumSGD(lr=1e-10, momentum=0.99)
    optimizer.setup(model)

    # train
    trainer = fcn.Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        weight_decay=0.0005,
        test_interval=1000,
        max_iter=10000,
        snapshot=1000,
        gpu=gpu,
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int,
                        help='if -1, use cpu only')
    args = parser.parse_args()
    main(args.gpu)
