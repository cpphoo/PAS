#!/bin/bash

# This script contains the commands used to train the supervised baseline and upper bound


# setup wandb and CUDA devices for training
wandb online
export CUDA_VISIBLE_DEVICES=0,1

#########################################################
# Training the supervised baseline on various datasests 
#########################################################

# Below are explanations to some of the relevant arguments
# --dir: where the models will be saved
# --trainpath: where the data could be found
# --label_file: a csv file specifying the coarse labels of each fine-grained classes used for training
# --label_space: a list of two strings. The first specifies the name of the coarse labels and the 
#                and the second specifies the name of the fine-grained labels 
# --remove_last_relu: remove the last relu of the last layer of the network. necessary for training 
#                     the network with cosine classifiers
# --use_cosine_clf: use the cosine classifier for training the representation

# CIFAR100-CL
python supervised.py \
--dir baseline_cifar100/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
--image_transformations standard_cifar \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest


# iNat2019-CL
python supervised.py \
--dir baseline_inat2019/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest


# tieredImageNet-CL 
python supervised.py \
--dir baseline_tieredImageNet/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 90 \
--lr 0.1 \
--lr_decay_freq 30 \
--wd 0.0005 \
--save_freq 30 \
--print_freq 20 \
--seed 1 \
--resume_latest


#########################################################
# Training the upper bound on various datasests 
#########################################################

# CIFAR100-CL
python supervised.py \
--dir upper_bound_cifar100/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/all_60 \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base_novel_seen.csv \
--image_transformations standard_cifar \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

# iNat2019-CL
python supervised.py \
--dir upper_bound_inat2019/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/all_60 \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base_novel_seen.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

# tieredImageNet-CL
python supervised.py \
--dir upper_bound_tieredImageNet/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/all_60 \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base_novel_seen.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 90 \
--lr 0.1 \
--lr_decay_freq 30 \
--wd 0.0005 \
--save_freq 30 \
--print_freq 20 \
--seed 1 \
--resume_latest


#########################################################
# Optional: Training the supervised baseline on various datasests with linear classifier
#           This is needed for running FEAT
#########################################################

# tieredImageNet-CL - linear classifier
python supervised.py \
--dir baseline_tieredImageNet_linear/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--bsize 256 \
--epochs 90 \
--lr 0.1 \
--lr_decay_freq 30 \
--wd 0.0005 \
--save_freq 30 \
--print_freq 20 \
--seed 1 \
--resume_latest

# CIFAR100-CL - linear classifier
python supervised.py \
--dir baseline_cifar100_linear/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
--image_transformations standard_cifar \
--num_workers 4 \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest


# iNat2019-CL - linear classifier
python supervised.py \
--dir baseline_inat2019_linear/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--image_transformations standard \
--num_workers 4 \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

##########################################################################
# Optional: Training the supervised baselines for the analyses section 6.2
##########################################################################
# section 6.2.1
python supervised.py \
--dir baseline_inat2019_base_reduced/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base_reduced.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest


# section 6.2.2
python supervised.py \
--dir baseline_inat2019_base_dropped_a_fifth_parent/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base_dropped_a_fifth_parent.csv \
--image_transformations standard \
--num_workers 4 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest

