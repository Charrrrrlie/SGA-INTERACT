## Experiment Environment
- CentOS Linux release 7.9.2009 (Core)
- Slurm Workload Manager
- CUDA 11.8
- Python 3.8
- PyTorch 2.2.2

## Installation
Modify the path in `scripts/train.sh` to the following \<PATH> if you run with shell.
```
conda create -p <PATH>/tactic python=3.8
conda activate tactic
```

`pytorch`

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

`other requirements`
```
pip install -r requirements.txt

# for GAL eval
pip install pandas
pip install joblib

----- Optional -----
# for flop counting
pip install fvcore

# for GAR confusion matrix
pip install seaborn
```