#!/usr/bin/env bash

python cifar.py -a resnet --depth 164 --block-name Bottleneck --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-164-bottleneck --gpu-id 0 &

python cifar.py -a resnet --depth 110 --block-name BasicBlock --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-basicblock --gpu-id 2 &

python cifar.py -a resnet --depth 110 --block-name Bottleneck --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-bottleneck --gpu-id 3 &

sleep infinity