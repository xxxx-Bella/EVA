# EVA
This repo contains the Pytorch implementation of our paper: 
> [**Evolution-aware VArance (EVA) Coreset Selection for Medical Image Classification**](https://arxiv.org/pdf/2406.05677.pdf)
>
> Yuxin Hong, Xiao Zhang, Xin Zhang, Joey Tianyi Zhou.

- **Accepted at ACMMM 2024 as Oral Presentation.**
- [**Nominated for Best Paper Awards**](https://2024.acmmm.org/best-paper).

![pipeline](pipeline.png)



# Installation and Requirements

Setup the required Python environments:
    pip install requirements.txt

## **Dateset**

We validate the effectiveness of EVA mainly on [MedMNIST](https://medmnist.com/). 

Then install `medmnist` as a standard Python package from [PyPI](https://pypi.org/project/medmnist/):

    pip install medmnist

Or install from source:

    pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Check whether you have installed the latest code [version](medmnist/info.py#L1):

    >>> import medmnist
    >>> print(medmnist.__version__)

Please download `.npz` files from [here](https://zenodo.org/records/10519652) to the `/data` directory. 
Then move/copy them to `/.medmnist`.


# **Getting Started**

## Train classifiers with the entire dataset
This step is **necessary** to collect training dynamics for future coreset selection.

    python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch_size 256 --task_name all-data --base_dir ./data-model/cifar10



**Coming Soon...**
