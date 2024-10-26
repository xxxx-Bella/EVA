import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys
import argparse
import pickle

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
from core.utils import print_training_info, StdRedirect

from torchvision.models import resnet18, resnet50                   
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny', 'svhn', 'cinic10', 'organamnist', 'organsmnist', 'tissuemnist', 'dermamnist', 'pneumoniamnist', 'pathmnist'])
parser.add_argument('--download', action="store_true")
parser.add_argument('--as_rgb', help='convert the grayscale image to RGB', action="store_true")
parser.add_argument('--e_min', type=int, help="early min epoch")
parser.add_argument('--e_max', type=int, help="early max epoch")
parser.add_argument('--l_min', type=int, help="later min epoch")
parser.add_argument('--l_max', type=int, help="later max epoch")
parser.add_argument('--wt', type=float, help="weight of the 1st window", default=0.5)



######################### Path Setting #########################
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base_dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task_name', type=str, default='tmp',
                    help='The name of the training task (all-data).')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

args = parser.parse_args()


######################### Set path variable #########################
e_min = args.e_min
e_max = args.e_max
l_min = args.l_min
l_max = args.l_max
wt = args.wt
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}-{e_min}-{e_max}-{l_min}-{l_max}.pickle')


######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
if dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif dataset == 'cifar100':
    num_classes=100
else:
    info = INFO[args.dataset]
    download = args.download
    as_rgb = args.as_rgb
    num_classes = len(info['label'])



def EVA(td_log, dataset, data_importance, e_min=0, e_max=10, l_min=100, l_max=110):
    print("\nCalculating EVA Score...")
    targets = []
    data_size = len(dataset)
    num_early_epochs = e_max - e_min
    num_later_epochs = l_max - l_min

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets) 
    targets = targets.squeeze()   
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['eva'] = torch.zeros(data_size).type(torch.float32)
    data_importance['early'] = torch.zeros((num_early_epochs, data_size)).type(torch.float32) 
    data_importance['later'] = torch.zeros((num_later_epochs, data_size)).type(torch.float32)

    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(item, epoch_idx, phase): 
        output = torch.exp(item['output'].type(torch.float))
        index = item['idx'].type(torch.long)
        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)

        score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1)) 

        if phase == 'early':
            data_importance['early'][epoch_idx-e_min, index] = score 
        else:
            data_importance['later'][epoch_idx-l_min, index] = score
    
    early_epoches = []
    later_epoches = []
    print(f'td_log length: {len(td_log)}')
    for i, item in enumerate(td_log):
        epoch_idx = item['epoch']  
        
        if i % 10000 == 0:
            print('EVA, td_log', i)
        
        # Record training dynamics of current epoch
        if (epoch_idx >= e_min) and (epoch_idx < e_max): 
            early_epoches.append(epoch_idx)
            record_training_dynamics(item, epoch_idx, 'early')
        elif (epoch_idx >= l_min) and (epoch_idx < l_max): 
            later_epoches.append(epoch_idx)
            record_training_dynamics(item, epoch_idx, 'later')
        if epoch_idx == l_max:
            print('early_epoches:', len(early_epoches))
            print('later_epoches:', len(later_epoches))

        if epoch_idx == e_max:  
            el2n_var_e = torch.var(data_importance['early'], dim=0)
            el2n_var_e_norm = F.normalize(el2n_var_e.unsqueeze(0), p=1, dim=1)
        elif epoch_idx == l_max: 
            el2n_var_l = torch.var(data_importance['later'], dim=0)
            el2n_var_l_norm = F.normalize(el2n_var_l.unsqueeze(0), p=1, dim=1)

            el2n_var = torch.squeeze(wt*el2n_var_e_norm + (1-wt)*el2n_var_l_norm)
            data_importance['eva'] = el2n_var
            
            return



GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([transforms.ToTensor(),])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}')

if 'mnist' in dataset:
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])
    trainset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
elif dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
elif dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)


trainset = IndexDataset(trainset)
print(len(trainset))

data_importance = {}

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=True)  # add drop_last

model = resnet('resnet18', num_classes=num_classes, device=device)
model = model.to(device)


print(f'Ckpt path: {ckpt_path}.')
checkpoint = torch.load(ckpt_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)

training_dynamics = pickled_data['training_dynamics']  # td_log

EVA(training_dynamics, trainset, data_importance, e_min=e_min, e_max=e_max, l_min=l_min, l_max=l_max)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)