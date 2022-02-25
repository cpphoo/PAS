#!/bin/bash

# This script contains the commands used to train metaOptNet
wandb online
export CUDA_VISIBLE_DEVICES=0,1

python metaOptNet.py \
--dir metaOptNet_cifar100_5way_10shot/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
--image_transformations standard_cifar \
--num_workers 4 \
--way 5 \
--shot 10 \
--query 6 \
--epochs 60 \
--episodes_per_epoch 8000 \
--bsize 8 \
--lr 0.1 \
--weight_decay 5e-4 \
--mom 0.9 \
--eps 0.1 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

python metaOptNet.py \
--dir metaOptNet_inat2019_5way_10shot/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--way 5 \
--shot 10 \
--query 6 \
--epochs 60 \
--episodes_per_epoch 4000 \
--bsize 4 \
--lr 0.1 \
--weight_decay 5e-4 \
--mom 0.9 \
--eps 0.1 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

python metaOptNet.py \
--dir metaOptNet_tieredImageNet_5way_10shot/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--way 5 \
--shot 10 \
--query 6 \
--epochs 60 \
--episodes_per_epoch 4000 \
--bsize 4 \
--lr 0.1 \
--weight_decay 5e-4 \
--mom 0.9 \
--eps 0.1 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest
