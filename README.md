# CSC413 Team 52
Yu Hang Ian Jian and Salman Shahid

## Introduction

This is the code implementation for the CSC413 Final Project for Team 52. Most of the code for model training and applying Mixup is taken from https://github.com/facebookresearch/mixup-cifar10. The code for RandAugment and the common image transformations used in it is taken from https://github.com/ildoonet/pytorch-randaugment. Our contributions were to add a train-validation split, enable training on datasets with images with a variable number of channels besides just RGB, and run Mixup and RandAugment together.

## Requirements and Installation

To run, simply clone this repo and change directory into it. Then:

1. Install requirements using
```
pip install -r requirements.txt
```

2. Run using the command below. For reproducibility, we used the seed given below in our experiments. To use Mixup, specify the desired value of alpha, such as --alpha=1. To use RandAugment, specify the RandAugment flag and set the desired values of N and M, like so: --rand-augment --N=1 --M=5
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --lr=0.1 --seed=20170922 --decay=1e-4
```
