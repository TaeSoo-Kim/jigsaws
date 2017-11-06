
# Gesture TCN Experiments

### JIGSAWS, kinematic, gesture classification,  Knot_Tying, LOSO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | 0.0638 | 2.2665 | 0.535 | 0.5 | adam | 14943918 | split=1
ResTCN_classification | 8 | 0.3188 | 0.2093 | 0.478 | 0.5 | adam | 14943956 | split=2
ResTCN_classification | 8 | x | x | 0.494 | 0.5 | adam | 14943957 | split=3
ResTCN_classification | 8 | x | x | 0.462 | 0.5 | adam | 14943958 | split=4
ResTCN_classification | 8 | x | x | 0.459 | 0.5 | adam | 14943960 | split=5


### JIGSAWS, kinematic, gesture classification,  Knot_Tying, LOUO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | x | x | 0.514 | 0.5 | adam | 14943961 | split=1
ResTCN_classification | 8 | x | x | 0.404 | 0.5 | adam | 14943962 | split=2
ResTCN_classification | 8 | x | x | 0.418 | 0.5 | adam | 14943963 | split=3
ResTCN_classification | 8 | x | x | 0.630 | 0.5 | adam | 14943964 | split=4
ResTCN_classification | 8 | x | x | 0.460 | 0.5 | adam | 14943965 | split=5
ResTCN_classification | 8 | x | x | 0.481 | 0.5 | adam | 14943966 | split=6
ResTCN_classification | 8 | x | x | 0.429 | 0.5 | adam | 14943967 | split=7
ResTCN_classification | 8 | x | x | 0.349 | 0.5 | adam | 14943968 | split=8

### JIGSAWS, kinematic, gesture classification,  Needle_Passing, LOSO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | x | x | 0.491 | 0.5 | adam | 14947580 | split=1
ResTCN_classification | 8 | x | x | 0.551 | 0.5 | adam | 14947586 | split=2
ResTCN_classification | 8 | x | x | 0.431 | 0.5 | adam | 14947587 | split=3
ResTCN_classification | 8 | x | x | 0.483 | 0.5 | adam | 14947650 | split=4
ResTCN_classification | 8 | x | x | 0.459 | 0.5 | adam | 14947651 | split=5


### JIGSAWS, kinematic, gesture classification,  Needle_Passing, LOUO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | x | x | 0.478 | 0.5 | adam | 14947653 | split=1
ResTCN_classification | 8 | x | x | 0.458 | 0.5 | adam | 14947654 | split=2
ResTCN_classification | 8 | x | x | 0.255 | 0.5 | adam | 14947655 | split=3
ResTCN_classification | 8 | x | x | 0.403 | 0.5 | adam | 14947659 | split=4
ResTCN_classification | 8 | x | x | 0.556 | 0.5 | adam | 14947660 | split=5
ResTCN_classification | 8 | x | x | x | 0.5 | adam | 14947665 | split=6, testlen = 0?
ResTCN_classification | 8 | x | x | 0.500 | 0.5 | adam | 14947668 | split=7
ResTCN_classification | 8 | x | x | 0.507 | 0.5 | adam | 14947670 | split=8

### JIGSAWS, kinematic, gesture classification,  Suturing, LOSO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | x | x | 0.426 | 0.5 | adam | 14947748 | split=1
ResTCN_classification | 8 | x | x | 0.459 | 0.5 | adam | 14947750 | split=2
ResTCN_classification | 8 | x | x | 0.437 | 0.5 | adam | 14947756 | split=3
ResTCN_classification | 8 | x | x | 0.487 | 0.5 | adam | 14947762 | split=4
ResTCN_classification | 8 | x | x | 0.429 | 0.5 | adam | 14947766 | split=5


### JIGSAWS, kinematic, gesture classification,  Suturing, LOUO
Model | filter | Training Loss | Testing Loss | Validation Acc |  Dropout | Opti | SLURM ID| Notes |
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---:  |
ResTCN_classification | 8 | x | x | 0.471 | 0.5 | adam | 14947767 | split=1
ResTCN_classification | 8 | x | x | 0.635 | 0.5 | adam | 14947775 | split=2
ResTCN_classification | 8 | x | x | 0.254 | 0.5 | adam | 14947785 | split=3
ResTCN_classification | 8 | x | x | 0.424 | 0.5 | adam | 14947787 | split=4
ResTCN_classification | 8 | x | x | 0.356 | 0.5 | adam | 14947790 | split=5
ResTCN_classification | 8 | x | x | 0.629 | 0.5 | adam | 14947792 | split=6, testlen = 0?
ResTCN_classification | 8 | x | x | 0.372 | 0.5 | adam | 14947793 | split=7
ResTCN_classification | 8 | x | x | 0.588 | 0.5 | adam | 14947799 | split=8
