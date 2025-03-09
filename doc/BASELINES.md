## Overview
The SGA-INTERACT benchmark includes group activity recognition (GAR) and temporal group activity localization (TGAL) tasks. 

The configuration files are separately stored by the task.

Several baseline methods are provided:
- Skeleton-based methods
    - [COMPOSER](https://github.com/hongluzhou/composer)
    - [MPGCN](https://github.com/mgiant/MP-GCN/blob/main/README.md)
- RGB-based methods from [DIN](https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark)
    - ARG
    - AT
    - DIN
    - STAtt (our baseline)


## Getting Started
### Data Preparation
Download the dataset and put it in `./data` folder. 

Download ST-GCN pretrained weight `ntuxsub-st_gcn.pt` from [here](https://drive.google.com/drive/folders/1y8HQuWTQcXS2gYz0xX5wOSQZPUdEnC17) and put it in `./data/stgcn_ckpt` folder. 

The structure is as follows:
```
data
├── basketball
├── stgcn_ckpt
└── ...
```

> [!NOTE]
> If you want to reproduce the results in the Volleyball dataset, we highly recommend you to refer the data preparation in [MPGCN](https://github.com/mgiant/MP-GCN/blob/main/README.md). <br>
> Step1: Download data from [COMPOSER](https://github.com/hongluzhou/composer)<br>
> Step2: Modify the path in `config/GAR/MPGCN/volleyball_gendata.yaml` and run data generation [script](https://drive.google.com/file/d/1GX5zpUAKbkn6e7iqv6UMxePAyyePTSV3/view?usp=drive_link)<br>
> Step3: Check the data structure 
```
data
├── volleyball
├── mpgcn_volleyball
└── ...
```
#

### Run Baselines
We provide scripts for training and testing on SLURM. You can manually launch the task using torchrun in the script.

Tensorboard records and training information will be saved in `log` directory.
```bash
cd scripts

./launch_train.sh <partition> <gpu_num> ../config/<config> <extra_tag(optional)>
```
Evaluation will be automatically launched after training for each epoch and record the best model.

# 
You can also manually launch the evaluation. 

Visualization will be the reward for your labor (lol).

For GAR, it will draw the confusion matrix (assert N_GPU=1) and for TGAL, it will plot the mAP curve.
```bash
torchrun --nproc-per-node <N_GPU> main.py --eval --config <config_path> --checkpoint log/<log_dir>/best.pth --extra_tag <extra_tag(Optional)>
```
 
Model FLOPs can be calculated by adding `--cal_flops` flag.

# 
NOTE: `model_params.time_embed.max_times_embed` in One2Many framework can be smaller (i.e. 100 for GAR and 200 for TGAL).
Removing current redundant time embedding may lead to better performance.
