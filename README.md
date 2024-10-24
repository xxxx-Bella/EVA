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

## Dateset

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


# Getting Started

## Train classifiers on the Entire Dataset
This step is **necessary** to collect training dynamics for future coreset selection.

    python train.py --dataset organamnist --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch_size 256 --task_name all-data --base_dir ./data-model/organamnist/resnet18 --download --as_rgb
    
After completing this step, you will obtain three `.npy` files for each epoch under the path `./data-model/all-data`.

## Sample importance score calculation
We need to first calcualte the different importance scores for coreset selection.

    python generate_importance_score.py --dataset organamnist --gpuid 0 --base_dir ./data-model/organamnist/resnet18 --e_min 0 --e_max 10 --l_min 100 --l_max 110 --task_name all-data --as_rgb

After the calculation, you will obtain a `.pickle` file storing sorted sample indexes and their respective importance scores.

## Train classifiers on the Selected Coreset
Here we use 30% selection rate on OrganAMNIST as an example.
    
    python train.py --dataset organamnist --gpuid 0 --epochs 200 --base_dir ./data-model/organamnist/eva --coreset --coreset_mode coreset --data-score-path ./data-model/organamnist/all-data/data-score-all-data-0-10-100-110.pickle --coreset_key eva --data-score-descending 1 --as_rgb --task_name eva-0-10-100-110-0.3  --coreset_ratio 0.3

* For an aggressive low selection rate, pls set a smaller batch size for better performance.


# Citation
If you find our work useful for your research, please cite our paper. 
    @inproceedings{hong2024EVA,
      title={Evolution-aware VArance (EVA) Coreset Selection for Medical Image Classification},
      author={Yuxin, Hong and Xiao, Zhang and Xin, Zhang and Joey Tianyi Zhou},
      booktitle={Proceedings of the 32st ACM International Conference on Multimedia},
      year={2024},
    }




