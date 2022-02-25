# Coarsely-labeled Data for Better Few-shot Transfer

## Introduction
This repo contains the official implementation of the following ICCV2021 paper:

**Title:** Coarsely-labeled Data for Better Few-shot Transfer  
**Authors:** Cheng Perng Phoo, Bharath Hariharan  
**Institution:** Cornell University  
**Paper:** https://openaccess.thecvf.com/content/ICCV2021/papers/Phoo_Coarsely-Labeled_Data_for_Better_Few-Shot_Transfer_ICCV_2021_paper.pdf  
**Abstract:**  
Few-shot learning is based on the premise that labels are expensive, especially when they are fine-grained and require expertise. But coarse labels might be easy to acquire and thus abundant. We present a representation learning approach - PAS that allows few-shot learners to leverage coarsely-labeled data available before evaluation. Inspired by self-training, we label the additional data using a teacher trained on the base dataset and filter the teacher's prediction based on the coarse labels; a new student representation is then trained on the base dataset and the pseudo-labeled dataset. PAS is able to produce a representation that consistently and significantly outperforms the baselines in 3 different datasets

### Requirements
This codebase is tested with:  
1. PyTorch 1.7.1
2. Torchvision 0.8.2
3. NumPy 
4. Pandas
5. wandb (for tracking purposes)


## Running Experiments 
### Step 1: Dataset Preparation
Download the dataset from google drive: https://drive.google.com/file/d/1H42xCS2I8LUW7JUZpAAPQEEw5I6c1q3J/view?usp=sharing

### Step 2: Training and Evaluating Representations/models
Models used in the paper are organized into different directories (we name different directories based on the model names). Training and evaluating models follow the following steps:
1. Go into the respective directory
2. To train models, look into `run.sh`, select the settings/datasets desired, and run the shell script. 
3. To evaluate the trained models, look into `run_evaluate.sh` in the same directory, select the appropriate settings, and then run the script. 

Note that the training script for  `Upper Bound` could be found in the `supervised_baseline/` directory whereas the code for `PAS` could be found in the `self_training/` directory.


## Notes 
1. Parts of the codebase are modified based on https://github.com/kjunelee/MetaOptNet and https://github.com/Sha-Lab/FEAT. We thank the authors for releasing their code to the public. 
2. Model checkpoints can be found here: https://drive.google.com/file/d/10UTXnnlPcGHrKATmMGqrsUMxuWm1Msvb/view?usp=sharing
3. If you find this codebase or PAS useful, please consider citing our paper: 
```
@inproceeding{phoo2021PAS,
    title={Coarsely-labeled Data for Better Few-shot Transfer},
    author={Phoo, Cheng Perng and Hariharan, Bharath},
    booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
    year={2021}
}
```