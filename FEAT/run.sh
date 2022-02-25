#!/bin/bash

# This script contains the commands used to train FEAT
# please make sure that the supervised models with linear classifier are available.
# these models will be loaded at the location specified using --pretrained_embedding_weights
# See run.sh in supervised_baseline/
wandb online
export CUDA_VISIBLE_DEVICES=0


for shot in 1 5
do 
    python FEAT.py \
    --dir FEAT_inat2019_5way_$shot\shot/resnet18_seed_1_wd_0.0005 \
    --trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
    --label_space genus id \
    --label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
    --image_transformations standard \
    --num_workers 4 \
    --way 5 \
    --shot $shot \
    --query 15 \
    --epochs 200 \
    --episodes_per_epoch 100 \
    --bsize 1 \
    --balance 0.1 \
    --temperature 32 \
    --temperature2 64 \
    --lr 0.0002 \
    --lr_mul 10 \
    --lr_scheduler step \
    --step_size 40 \
    --gamma 0.5 \
    --use_euclidean \
    --weight_decay 5e-4 \
    --save_freq 50 \
    --print_freq 20 \
    --seed 1 \
    --resume_latest \
    --pretrained_embedding_weights ../supervised_baseline/baseline_inat2019_linear/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl

    python FEAT.py \
    --dir FEAT_cifar100_5way_$shot\shot/resnet18_seed_1_wd_0.0005 \
    --trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
    --label_space parent id \
    --label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
    --image_transformations standard_cifar \
    --num_workers 4 \
    --way 5 \
    --shot $shot \
    --query 15 \
    --epochs 200 \
    --episodes_per_epoch 100 \
    --bsize 1 \
    --balance 0.1 \
    --temperature 32 \
    --temperature2 64 \
    --lr 0.0002 \
    --lr_mul 10 \
    --lr_scheduler step \
    --step_size 40 \
    --gamma 0.5 \
    --use_euclidean \
    --weight_decay 5e-4 \
    --save_freq 50 \
    --print_freq 20 \
    --seed 1 \
    --resume_latest \
    --pretrained_embedding_weights ../supervised_baseline/baseline_cifar100_linear/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl

    python FEAT.py \
    --dir FEAT_tieredImageNet_5way_$shot\shot/resnet18_seed_1_wd_0.0005 \
    --trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
    --label_space parent id \
    --label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
    --image_transformations standard \
    --num_workers 4 \
    --way 5 \
    --shot $shot \
    --query 15 \
    --epochs 200 \
    --episodes_per_epoch 100 \
    --bsize 1 \
    --balance 0.1 \
    --temperature 32 \
    --temperature2 64 \
    --lr 0.0002 \
    --lr_mul 10 \
    --lr_scheduler step \
    --step_size 40 \
    --gamma 0.5 \
    --use_euclidean \
    --weight_decay 5e-4 \
    --save_freq 50 \
    --print_freq 20 \
    --seed 1 \
    --resume_latest \
    --pretrained_embedding_weights ../supervised_baseline/baseline_tieredImageNet_linear/resnet18_seed_1_wd_0.0005/checkpoint_90.pkl
done