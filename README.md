# Online Coreset Selection for Rehearsal-based Continual Learning

This repository is an official Pytorch implementation of [Online Coreset Selection for Rehearsal-based Continual Learning](https://openreview.net/forum?id=f9D-5WNG4Nv) (**ICLR 2022**)

<!-- > Currently working on PyTorch version  -->
<img align="middle" width="800" src="https://github.com/jaehong31/OCS/blob/main/OCS_concept.png">

## Abstract

A dataset is a shred of crucial evidence to describe a task. However, each data point in the dataset does not have the same potential, as some of the data points can be more representative or informative than others. This unequal importance among the data points may have a large impact in rehearsal-based continual learning, where we store a subset of the training examples (coreset) to be replayed later to alleviate catastrophic forgetting. In continual learning, the quality of the samples stored in the coreset directly affects the model's effectiveness and efficiency. The coreset selection problem becomes even more important under realistic settings, such as imbalanced continual learning or noisy data scenarios. To tackle this problem, we propose *Online Coreset Selection (OCS)*, a simple yet effective method that selects the most representative and informative coreset at each iteration and trains them in an online manner. Our proposed method maximizes the model's adaptation to a target dataset while selecting high-affinity samples to past tasks, which directly inhibits catastrophic forgetting. We validate the effectiveness of our coreset selection mechanism over various standard, imbalanced, and noisy datasets against strong continual learning baselines, demonstrating that it improves task adaptation and prevents catastrophic forgetting in a sample-efficient manner. 

__Contribution of this work__
- We address the problem of coreset selection for realistic and challenging continual learning, where the data continuum is composed of class-imbalanced or noisy instances that deteriorates the performance of the continual learner during training.
- We propose *Online Coreset Selection (OCS)*, a simple yet effective online coreset selection method to obtain a subset from each minibatch during continual learning, which is representative, diverse, and has high-affinity to the previous tasks. Specifically, we present three gradient-based selection criteria to select the coreset for current task adaptation and to alleviate catastrophic forgetting. 
- We demonstrate that OCS is applicable to any rehearsal-based continual learning method, and experimentally validate it on multiple benchmark scenarios, where it largely improves the performance of the base algorithms across various performance metrics.


<!-- ## Codes
The initial code is [Here](https://openreview.net/forum?id=f9D-5WNG4Nv), and we will provide the refactorized repository with sufficient explanations ASAP. Stay tuned. -->


## Prerequisites
```
$ pip install -r requirements.txt
```

## Run
1. __MNIST__ experiment
```
$ python ocs_mnist.py
```

2. __Split CIFAR-100__ experiment

```
$ python ocs_cifar.py
```

3. __Multiple Datasets__ experiment

```
$ python ocs_mixture.py
```



## Citations
```
@inproceedings{yoon2022online,
    title={Online Coreset Selection for Rehearsal-based Continual Learning},
    author={Jaehong Yoon and Divyam Madaan and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/pdf?id=f9D-5WNG4Nv}
}
```