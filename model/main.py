from multi_input_resnet import resnet18
from torchinfo import summary
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import argparse
from training import train_model


parser = argparse.ArgumentParser()
parser.add_argument('--initial_lr', type=float, help='initial learning rate', default=0.001)
parser.add_argument('--momentum', type=float, help='momentum', default=0.5)
parser.add_argument('--step_size', type=int, default=7)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--workers', type=int, default=2, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log-every-n-steps', default=1000, type=int,
                    help='Log every n steps')
parser.add_argument('--checkpoint', type=bool, default=False)
args = parser.parse_args()


PATH = './trained'
num_channels = 5
chunk_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18(device=device, input_channels=num_channels)
#print(model)
summary(model, input_size=(2, num_channels, chunk_size, chunk_size), col_names=['input_size', 'output_size', 'num_params'])

if __name__=="__main__":
    data = np.load("..\\dataset\\train_data_64.npy")
    targets = np.load("..\\dataset\\targets_64.npy")
    data = data[110**2 // 2:]
    targets = targets[110**2 // 2:]
    tensor_data = torch.Tensor(data)
    tensor_targets = torch.Tensor(targets)

    unique, counts = np.unique(targets, return_counts=True)
    print(unique, counts)
    class_weights = [1.0/c for c in counts]
    sample_weights = [class_weights[i] for i in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    dataset = TensorDataset(tensor_data,tensor_targets)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,  sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), args.initial_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    model = train_model(model, criterion, optimizer, scheduler, train_loader, args.epochs, device, args.batch_size, args)
    torch.save(model.state_dict(), PATH)





