Classification for Standford_Dogs


## Introduction
This is the code for Classification for Standford_Dogs, realized by SVM and deep metric learning.


## Quick Start
For deep metric model:
```Bash  
CUDA_VISIBLE_DEVICES=[] python train.py --exp_name 'margin' --code_size 64 --triplet_margin 8 --dataset_name 'Stanford_Dogs' --model resnet18
``` 

for svm:
```Bash  
python svm.py --c 0.5 --k 'rbf'
``` 