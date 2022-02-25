#!/bin/bash

# this script contains the commands used for 5-way evaluation for FEAT. 


export CUDA_VISIBLE_DEVICES=0
num_repetitions=10000

for model in FEAT
do
    for dataset in inat2019 tieredImageNet cifar100
    do
        model_name=checkpoint_200.pkl

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
        
        echo Model Name: $model_name
        echo Dataset Directory: $dataset_directory
        echo Label Space: $label_space
        echo Image Transformation: $transformation

        for shot in 1 5
        do

            target_directory=$model\_$dataset\_5way_$shot\shot/resnet18_seed_1_wd_0.0005
            echo Model Directory: $target_directory

            python evaluate_FEAT.py \
            --ref_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
            --query_path ../few_shot_coarse_labels/$dataset_directory\/eval/ref \
            --novel_categories ../few_shot_coarse_labels/$dataset_directory\/novel.csv \
            --bsize 512 \
            --num_workers 4 \
            --result_file $target_directory\_novel_5_way_3_max_parent \
            --label_space $label_space \
            --image_transformations $transformation \
            --ckpt  $target_directory\/$model_name \
            --num_repetitions $num_repetitions \
            --shots $shot \
            --ways 5 \
            --num_max_parents_per_task 3 \
            --seed 1 

            python evaluation/compile_result.py --file $target_directory\_novel_5_way_3_max_parent.xlsx
        done 
    done
done