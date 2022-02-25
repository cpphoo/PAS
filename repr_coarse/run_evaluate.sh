#!/bin/bash

# This script contains the commands used to conduct all-way and 5-way nearest centroid evaluation for the supervised baseline and upper bound

export CUDA_VISIBLE_DEVICES=0
##########################
# All-way Evaluation
##########################
# set the number of rounds to be 1000

num_repetitions=1000

for model in repr_coarse
do
    for dataset in inat2019 tieredImageNet cifar100
    do

        target_directory=$model\_$dataset\/resnet18_seed_1_wd_0.0005

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            model_name=checkpoint_90.pkl
        else
            model_name=checkpoint_300.pkl
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            dataset_directory=tieredImageNet-CL
        elif [[ "$dataset" == "cifar100" ]]
        then
            dataset_directory=CIFAR-100-CL
        else
            dataset_directory=iNat2019-CL
        fi

        if [[ "$dataset" == "inat2019" ]]
        then 
            label_space="genus id"
        else
            label_space="parent id"
        fi

        if [[ "$dataset" == "cifar100" ]]
        then 
            transformation="standard_cifar"
        else
            transformation="standard"
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then 
            label_file_append="_keep"
        else
            label_file_append=""
        fi
        
        echo "*************" Starting Evaluation "*********************"
        echo Model Directory: $target_directory
        echo Model Name: $model_name
        echo Dataset Directory: $dataset_directory
        echo Label Space: $label_space
        echo Image Transformation: $transformation

        # evaluate all the novel classes including seen and unseen
        python evaluation/evaluate_nearest_centroid.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel \
        --cosine \
        --remove_last_relu \
        --label_space $label_space \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --shots 1 5 -1 \
        --seed 1 


        python evaluation/compile_result.py --file $target_directory\_novel.xlsx
        
        # evaluate novel seen classes
        python evaluation/evaluate_nearest_centroid.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel_seen$label_file_append\.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel_seen \
        --cosine \
        --remove_last_relu \
        --label_space $label_space \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --shots 1 5 -1 \
        --seed 1 


        python evaluation/compile_result.py --file $target_directory\_novel_seen.xlsx

        # evaluate novel unseen classes
        python evaluation/evaluate_nearest_centroid.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel_unseen$label_file_append\.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel_unseen \
        --cosine \
        --remove_last_relu \
        --label_space $label_space \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --shots 1 5 -1 \
        --seed 1 

        python evaluation/compile_result.py --file $target_directory\_novel_unseen.xlsx
    done
done

##########################
# 5-way Evaluation
##########################
num_repetitions=10000

for model in repr_coarse 
do
    for dataset in inat2019 tieredImageNet cifar100
    do

        target_directory=$model\_$dataset\/resnet18_seed_1_wd_0.0005

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            model_name=checkpoint_90.pkl
        else
            model_name=checkpoint_300.pkl
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            dataset_directory=tieredImageNet-CL
        elif [[ "$dataset" == "cifar100" ]]
        then
            dataset_directory=CIFAR-100-CL
        else
            dataset_directory=iNat2019-CL
        fi

        if [[ "$dataset" == "inat2019" ]]
        then 
            label_space="genus id"
        else
            label_space="parent id"
        fi

        if [[ "$dataset" == "cifar100" ]]
        then 
            transformation="standard_cifar"
        else
            transformation="standard"
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then 
            label_file_append="_keep"
        else
            label_file_append=""
        fi
        
        echo "*************" Starting Evaluation "*********************"
        echo Model Directory: $target_directory
        echo Model Name: $model_name
        echo Dataset Directory: $dataset_directory
        echo Label Space: $label_space
        echo Image Transformation: $transformation

        python evaluation/evaluate_nearest_centroid_Nway.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel_5_way_3_max_parent \
        --cosine \
        --remove_last_relu \
        --label_space $label_space \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --shots 1 5 \
        --ways 5 \
        --num_max_parents_per_task 3 \
        --seed 1 

        python evaluation/compile_result.py --file $target_directory\_novel_5_way_3_max_parent.xlsx
        
    done
done

###############################################
# Optional: NN-ext in the supplementary materials
################################################
num_repetitions=1000

for model in repr_coarse
do
    for dataset in inat2019
    do

        target_directory=$model\_$dataset\/resnet18_seed_1_wd_0.0005

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            model_name=checkpoint_90.pkl
        else
            model_name=checkpoint_300.pkl
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then
            dataset_directory=tieredImageNet-CL
        elif [[ "$dataset" == "cifar100" ]]
        then
            dataset_directory=CIFAR-100-CL
        else
            dataset_directory=iNat2019-CL
        fi

        if [[ "$dataset" == "inat2019" ]]
        then 
            label_space="genus id"
        else
            label_space="parent id"
        fi

        if [[ "$dataset" == "cifar100" ]]
        then 
            transformation="standard_cifar"
        else
            transformation="standard"
        fi

        if [[ "$dataset" == "tieredImageNet" ]]
        then 
            label_file_append="_keep"
        else
            label_file_append=""
        fi
        
        echo "*************" Starting Evaluation "*********************"
        echo Model Directory: $target_directory
        echo Model Name: $model_name
        echo Dataset Directory: $dataset_directory
        echo Label Space: $label_space
        echo Image Transformation: $transformation

        # NN-ext with the common parent
        python evaluation/evaluate_nearest_centroid_extension.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel.csv \
        --label_space $label_space \
        --base_path ../few_shot_coarse_labels/$dataset_directory\/repr \
        --base_label_space $label_space \
        --base_label_file ../few_shot_coarse_labels/$dataset_directory\/base.csv \
        --coarsely_labeled_path ../few_shot_coarse_labels/$dataset_directory\/novel_60 \
        --coarsely_labeled_label_space $label_space \
        --coarsely_labeled_label_file ../few_shot_coarse_labels/$dataset_directory\/novel_seen.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel_NN_ext \
        --cosine \
        --remove_last_relu \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --seed 1 \
        --shots 1 5 -1 \
        --num_candidates 100 \
        --num_nearest_neighbors 10 \
        --same_parent 

        python evaluation/compile_result.py --file $target_directory\_novel_NN_ext.xlsx

        # NN-ext without any restriction
        python evaluation/evaluate_nearest_centroid_extension.py \
        --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
        --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel.csv \
        --label_space $label_space \
        --base_path ../few_shot_coarse_labels/$dataset_directory\//repr \
        --base_label_space $label_space \
        --base_label_file ../few_shot_coarse_labels/$dataset_directory\/base.csv \
        --coarsely_labeled_path ../few_shot_coarse_labels/$dataset_directory\/novel_60 \
        --coarsely_labeled_label_space $label_space \
        --coarsely_labeled_label_file ../few_shot_coarse_labels/$dataset_directory\/novel_seen.csv \
        --bsize 512 \
        --num_workers 4 \
        --result_file $target_directory\_novel_NN_ext_Any \
        --cosine \
        --remove_last_relu \
        --image_transformations $transformation \
        --ckpt  $target_directory\/$model_name \
        --num_repetitions $num_repetitions \
        --seed 1 \
        --shots 1 5 -1 \
        --num_candidates 100 \
        --num_nearest_neighbors 10 

        python evaluation/compile_result.py --file $target_directory\_novel_NN_ext_Any.xlsx
    done
done