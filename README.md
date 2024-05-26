# RCL_main

Pytorch implementation for "**Improving Data-aware and Parameter-aware Robustness for Continual Learning**"

## Abstract
The goal of Continual Learning (CL) tasks is to continuously learn multiple new tasks sequentially while achieving a balance between the plasticity and stability of new and old knowledge. This paper analyzes that this insufficiency arises from the ineffective handling of outliers, leading to abnormal gradients and unexpected model updates. To address this issue, we enhance the data-aware and parameter-aware robustness of CL, proposing a Robust Continual Learning (RCL) method. From the data perspective, we develop a contrastive loss based on the concepts of uniformity and alignment, forming a feature distribution that is more applicable to outliers. From the parameter perspective, we present a forward strategy for worst-case perturbation and apply robust gradient projection to the parameters. The experimental results on three benchmarks show that the proposed method effectively maintains robustness and achieves new state-of-the-art (SOTA) results.

## Experiments

### Datasets

The datasets for CIFAR-100 and 5-Datasets will be automatically downloaded. For the experiments on MiniImageNet, please download the [train_data](https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view?usp=sharing) and [test_data](https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view?usp=sharing).

### Code

Please configure the path of the data set in dataloader first.

Run CIFAR100
> python main_rcl_cifar100.py

Run Five Datasets
> python main_rcl_fivedataset.py

Run MiniImagenet
> python main_rcl_miniimagenet.py


Tip: The default hyperparameters in the main_rcl_xxx.py file are not necessarily the optimal hyperparameters. You can further check the hyperparameter configuration in our **./logs/xxx/log_date.txt** to reproduce the results.


### Acknowledgement
Our implementation references the code below, thanks to them.

[EnnengYang/DFGP](https://github.com/EnnengYang/DFGP), [SsnL/align_uniform](https://github.com/SsnL/align_uniform)
