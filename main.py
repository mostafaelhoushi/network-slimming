from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import copy
from contextlib import redirect_stdout

import bnutils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--freeze-weights', dest='freeze_weights', action='store_true',
                    help='freeze weights of convolution and fully-connected layers')
parser.add_argument('--freeze-biases', dest='freeze_biases', action='store_true',
                    help='freeze biases of convolution and fully-connected layers')
parser.add_argument('--freeze-gamma', dest='freeze_gamma', action='store_true',
                    help='freeze gamma of batchnorm layers')
parser.add_argument('--freeze-beta', dest='freeze_beta', action='store_true',
                    help='freeze beta of batchnorm layers')
parser.add_argument('--no-backward-pass', dest='no_backward_pass', action='store_true',
                    help='during training just do forward pass to update the running mean and var of batchnorm layers')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                   help='evaluate only')
parser.add_argument('--update-mean-var', dest='update_mean_var', action='store_true',
                    help='replace running_mean and running_var of batchnorm layers with absolute mean and var')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--print-weights', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='For printing the weights of Model (default: True)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.freeze_weights:
    model = bnutils.freeze_weights(model)
if args.freeze_biases:
    model = bnutils.freeze_biases(model)
if args.freeze_gamma:
    model = bnutils.freeze_gamma(model)
if args.freeze_beta:
    model = bnutils.freeze_beta(model)
if args.update_mean_var:
    model = bnutils.update_mean_var(model, train_loader) 

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        if args.no_backward_pass is False:
            loss.backward()
            if args.sr:
                updateBN()
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

        train_loss += loss.data.item()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= len(train_loader.dataset)
    acc = 100. * correct.cpu().numpy() / float(len(train_loader.dataset))
    epoch_train_time = time.time() - end
    batch_train_avg_time = epoch_train_time / float(len(train_loader.dataset))

    return (train_loss, acc, batch_train_avg_time)    

def test():
    model.eval()
    test_loss = 0
    correct = 0
    end = time.time()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct.cpu().numpy() / float(len(test_loader.dataset)) 
    epoch_test_time = time.time() - end
    batch_test_avg_time = epoch_test_time / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return (test_loss, acc, batch_test_avg_time)

def save_checkpoint(state, is_best, dir_path):
    torch.save(state, os.path.join(dir_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, 'checkpoint.pth.tar'), os.path.join(dir_path, 'model_best.pth.tar'))

    if (state['epoch']-1)%10 == 0:
        os.makedirs(os.path.join(dir_path, 'checkpoints'), exist_ok=True)
        shutil.copyfile(os.path.join(dir_path, 'checkpoint.pth.tar'), os.path.join(dir_path, 'checkpoints', 'checkpoint_' + str(state['epoch']-1) + '.pth.tar'))    

model_dir = args.save
best_prec1 = 0.
# initialize log
train_log = []
start_log_time = time.time()
with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
    train_log_csv = csv.writer(train_log_file)
    train_log_csv.writerow(['epoch', 'train_loss', 'train_top1_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_time', 'cumulative_time'])

if args.evaluate:
    val_epoch_log = test()
    (_, prec1, _) = val_epoch_log
    best_prec1 = prec1
else:
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train_epoch_log = train(epoch)
        val_epoch_log = test()
        (_, prec1, _) = val_epoch_log
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (args.print_weights):
            os.makedirs(os.path.join(model_dir, 'weights_logs'), exist_ok=True)
            with open(os.path.join(model_dir, 'weights_logs', 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
                with redirect_stdout(weights_log_file):
                    # Log model's state_dict
                    print("Model's state_dict:")
                    # TODO: Use checkpoint above
                    for param_tensor in model.state_dict():
                        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                        print(model.state_dict()[param_tensor])
                        print("")

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, dir_path=model_dir)

        if is_best:
            torch.save(model.state_dict(), os.path.join(model_dir, "weights.pth"))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pth"))
            torch.save(model, os.path.join(model_dir, "model.pth"))


        # append to log
        with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
            train_log_csv = csv.writer(train_log_file)
            train_log_csv.writerow(((epoch,) + train_epoch_log + val_epoch_log + (time.time() - start_log_time,))) 


print("Best accuracy: "+str(best_prec1))
