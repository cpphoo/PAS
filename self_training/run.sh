#!/bin/bash

# This script contains the commands used to train the self-training baseline and PAS
wandb online
export CUDA_VISIBLE_DEVICES=0,1

#########################################################
# Training PAS on various datasests 
# Please make sure that the supervised baseline are available for generating the pseudolabels 
#########################################################

# Below are explanations to some of the relevant arguments
# --dir: where the models will be saved
# --trainpath: where the data could be found
# --label_file: a csv file specifying the coarse labels of each fine-grained classes used for training
# --label_space: a list of two strings. The first specifies the name of the coarse labels and the 
#                and the second specifies the name of the fine-grained labels 
# --secondary_trainpath: where the coarsely-labeled data could be found
# --secondary_label_file: a csv file specifying the coarse labels for the coarsely-labeled data.
# --secondary_label_space: a list of two strings. The first specifies the name of the coarse labels and the 
#                and the second specifies the name of the fine-grained labels 
# --remove_last_relu: remove the last relu of the last layer of the network. necessary for training 
#                     the network with cosine classifiers
# --use_cosine_clf: use the cosine classifier for training the representation
# --teacher_state_dict: location of the teacher model for generating the pseudolabels 
# --parent_consistent: whether to perform parent consistent filtering for generating the pseudolabels for training
#                       Turning this on would yield the PAS representation. Otherwise, the model is a self-training model

# CIFAR100-CL
python self_training.py \
--dir PAS_cifar100/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/CIFAR-100-CL/novel_60 \
--secondary_label_space parent id \
--secondary_label_file ../few_shot_coarse_labels/CIFAR-100-CL/novel_seen.csv \
--image_transformations standard_cifar \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_cifar100/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl
--parent_consistent \

# iNat2019-CL
python self_training.py \
--dir PAS_inat2019/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/novel_60 \
--secondary_label_space genus id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_inat2019/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl
--parent_consistent \

# tieredImageNet-CL
python self_training.py \
--dir PAS_multi_tieredImageNet/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/tieredImageNet-CL/novel_60 \
--secondary_label_space parent id \
--secondary_label_file ../few_shot_coarse_labels/tieredImageNet-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 90 \
--lr 0.1 \
--lr_decay_freq 30 \
--wd 0.0005 \
--save_freq 30 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_tieredImageNet/resnet18_seed_1_wd_0.0005/checkpoint_90.pkl
--parent_consistent \


#########################################################
# Training self-training representation on various datasests 
# Please make sure that the supervised baseline are available for generating the pseudolabels 
#########################################################
# CIFAR100-CL
python self_training.py \
--dir self_training_cifar100/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/CIFAR-100-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/CIFAR-100-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/CIFAR-100-CL/novel_60 \
--secondary_label_space parent id \
--secondary_label_file ../few_shot_coarse_labels/CIFAR-100-CL/novel_seen.csv \
--image_transformations standard_cifar \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_cifar100/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl


# iNat2019-CL
python self_training.py \
--dir self_training_inat2019/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/novel_60 \
--secondary_label_space genus id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_inat2019/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl

# tieredImageNet-CL
python self_training.py \
--dir self_training_multi_tieredImageNet/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/tieredImageNet-CL/repr \
--label_space parent id \
--label_file ../few_shot_coarse_labels/tieredImageNet-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/tieredImageNet-CL/novel_60 \
--secondary_label_space parent id \
--secondary_label_file ../few_shot_coarse_labels/tieredImageNet-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 90 \
--lr 0.1 \
--lr_decay_freq 30 \
--wd 0.0005 \
--save_freq 30 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--teacher_state_dict ../supervised_baseline/baseline_tieredImageNet/resnet18_seed_1_wd_0.0005/checkpoint_90.pkl


#########################################################
# Optional:  Filtering with different coarse labels (section 6.2.3)
#########################################################
# PAS with the order level filtering
python self_training.py \
--dir PAS_inat2019_order/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space order id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/novel_60 \
--secondary_label_space order id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--parent_consistent \
--teacher_state_dict ../supervised_baseline/baseline_inat2019/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl


# PAS with kingdom level filtering
python self_training.py \
--dir PAS_inat2019_kingdom/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space kingdom id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/novel_60 \
--secondary_label_space kingdom id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--parent_consistent \
--teacher_state_dict ../supervised_baseline/baseline_inat2019/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl


##########################################################################
# Optional: Training PAS for the analyses section 6.2
##########################################################################
# section 6.2.1
python self_training.py \
--dir PAS_inat2019_base_reduced/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base_reduced.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/all_60 \
--secondary_label_space genus id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen_extended.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--parent_consistent \
--teacher_state_dict ../supervised_baseline/baseline_inat2019_base_reduced/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl

# section 6.2.2
python self_training.py \
--dir PAS_inat2019_base_dropped_a_fifth_parent/resnet18_seed_1_wd_0.0005 \
--trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
--label_space genus id \
--label_file ../few_shot_coarse_labels/iNat2019-CL/base_dropped_a_fifth_parent.csv \
--secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/all_60 \
--secondary_label_space genus id \
--secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
--image_transformations standard \
--num_workers 5 \
--remove_last_relu \
--use_cosine_clf \
--bsize 256 \
--secondary_bsize 128 \
--epochs 300 \
--lr 0.1 \
--lr_decay_freq 100 \
--wd 0.0005 \
--save_freq 50 \
--print_freq 20 \
--seed 1 \
--resume_latest \
--parent_consistent \
--teacher_state_dict ../supervised_baseline/baseline_inat2019_base_dropped_a_fifth_parent/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl


##########################################################################
# Optional: Running PAS with different amounts of coarsely-labeled data (supplementary materials)
##########################################################################
for coarse_ratio in 0.05 0.1 0.25 0.5 0.75
do
    python self_training.py \
    --dir PAS_inat2019_coarse_ratio_$coarse_ratio\/resnet18_seed_1_wd_0.0005 \
    --trainpath ../few_shot_coarse_labels/iNat2019-CL/repr \
    --label_space genus id \
    --label_file ../few_shot_coarse_labels/iNat2019-CL/base.csv \
    --secondary_trainpath ../few_shot_coarse_labels/iNat2019-CL/novel_60 \
    --secondary_label_space genus id \
    --secondary_label_file ../few_shot_coarse_labels/iNat2019-CL/novel_seen.csv \
    --image_transformations standard \
    --num_workers 5 \
    --remove_last_relu \
    --use_cosine_clf \
    --bsize 256 \
    --secondary_bsize 128 \
    --epochs 300 \
    --lr 0.1 \
    --lr_decay_freq 100 \
    --wd 0.0005 \
    --save_freq 50 \
    --print_freq 20 \
    --seed 1 \
    --resume_latest \
    --parent_consistent \
    --teacher_state_dict ../supervised_baseline/baseline_inat2019/resnet18_seed_1_wd_0.0005/checkpoint_300.pkl \
    --coarse_label_ratio $coarse_ratio
done