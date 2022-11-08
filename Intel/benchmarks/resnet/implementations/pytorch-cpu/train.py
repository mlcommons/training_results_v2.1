import argparse
from calendar import EPOCH
import os
import random
import shutil
import time
import warnings
import threading
import types

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch  as ipex

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import resnet
from torch.utils import ThroughputBenchmark

import oneccl_bindings_for_pytorch

from lars import create_optimizer_lars, Lars
from utils import *
from val_sampler import *

from mlperf_logger import mllogger
from mlperf_logging.mllog.constants import (SUBMISSION_BENCHMARK, SUBMISSION_DIVISION, SUBMISSION_STATUS,
    SUBMISSION_ORG, SUBMISSION_PLATFORM,
    RESNET, CLOSED, ONPREM, EVAL_ACCURACY, STATUS, SUCCESS, ABORTED,
    INIT_START, INIT_STOP, RUN_START, RUN_STOP, SEED, GLOBAL_BATCH_SIZE, TRAIN_SAMPLES, MODEL_BN_SPAN,
    EVAL_SAMPLES, EPOCH_COUNT, FIRST_EPOCH_NUM, OPT_NAME, ADAM, LARS, OPT_BASE_LR, OPT_WEIGHT_DECAY,
    OPT_LR_WARMUP_EPOCHS, OPT_LR_WARMUP_FACTOR, GRADIENT_ACCUMULATION_STEPS, EPOCH_START, EPOCH_STOP,
    EVAL_START, EVAL_STOP, BLOCK_START, BLOCK_STOP)

model_names = ['resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-checkpoint', action='store_true', default=False,
                    help='Save checkpoint after validation (default: none)')                    

# Distributed training args
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='ccl', type=str,
                    help='distributed backend') 

# IPEX args
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use Intel Pytorch Extension')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable ipex bf16 path')

# Learning Hyperparams 
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')  
parser.add_argument('--base-op', type=str, default='sgd',
                        help='base optimizer name, supports SGD and LARS')                                      
parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='base learning rate for SGD and LARS')             
parser.add_argument('--end-lr', type=float, default=0.0001,
                        help='end learning rate for polynomial decay LR schedule')       
parser.add_argument('--poly-power', type=int, default=2,
                        help='power for polynomial decay LR schedule')                        
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')                     
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='global-batch size (default: 256), this is the total '
                         'batch size of all CPUs when using Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')             
parser.add_argument('--epsilon', type=float, default=0,
                        help='epsilon for optimizer')         
parser.add_argument('--bn-bias-separately', action='store_true', default=True,
                        help='skip bn and bias') 
parser.add_argument('--label-smoothing', type=float, default=0.1, 
                    help='label smoothing for cross entropy loss')        
parser.add_argument('--zero-init-residual', action='store_true', default=False,
                    help='Initialize scale params in BN3 of a residual block to zeros instead ones. '
                         'Improves accuracy by 0.2~0.3 percent according to https://arxiv.org/abs/1706.02677')
# Evaluation args
parser.add_argument('--target-acc', default=0.759, type=float, help='Target validation accuracy')
parser.add_argument('--target-epoch', default=35, type=float, help='Target number of epochs')
parser.add_argument('--eval-period', default=4, type=int, help='Evaluate every eval_period epochs')
parser.add_argument('--eval-offset', default=3, type=int, help="Start the first evaluation after eval_offset'th epoch")

best_acc1 = 0

def main():

    torchvision.set_image_backend('accimage')
    args = parser.parse_args()
    print(args)
    
    # CCL related 			
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', '127.0.0.1'))
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("World size: ", args.world_size)

    args.distributed = args.world_size > 1 
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    if args.seed is not None:
        if args.distributed:
            local_seed = args.seed + args.rank
            random.seed(local_seed)
            torch.manual_seed(local_seed)
        else:            
            random.seed(args.seed)
            torch.manual_seed(args.seed)    
    main_worker(args)

def main_worker(args):    
    
    global best_acc1

    mllogger.event(key="cache_clear", value=True) 
    mllogger.event(key=SUBMISSION_BENCHMARK, value=RESNET)
    mllogger.event(key=SUBMISSION_ORG, value="Intel")
    mllogger.event(key=SUBMISSION_DIVISION, value=CLOSED)
    mllogger.event(key=SUBMISSION_STATUS, value=ONPREM)
    mllogger.event(key=SUBMISSION_PLATFORM, value="Intel Xeon (R) codenamed Sapphire Rapids")
    mllogger.start(key=INIT_START, sync=True)

    mllogger.event(key=SEED, value=args.seed)
    mllogger.event(key=MODEL_BN_SPAN, value=args.batch_size/args.world_size)
    mllogger.event(key=GLOBAL_BATCH_SIZE, value=args.batch_size)

    mllogger.event(key=OPT_NAME, value=LARS)
    mllogger.event(key="lars_opt_base_learning_rate", value=args.base_lr)
    mllogger.event(key="lars_opt_learning_rate_warmup_epochs", value=args.warmup_epochs)
    mllogger.event(key="lars_opt_learning_rate_decay_steps", value=(args.epochs-args.warmup_epochs))
    mllogger.event(key="lars_opt_momentum", value=args.momentum)
    mllogger.event(key="lars_opt_weight_decay", value=args.weight_decay)
    mllogger.event(key="lars_opt_end_learning_rate", value=args.end_lr)
    mllogger.event(key="lars_opt_learning_rate_decay_poly_power", value=args.poly_power)
    mllogger.event(key="lars_epsilon", value=args.epsilon)
    mllogger.event(key="gradient_accumulation_steps", value=1) 

							
    print("=> Creating model '{}'".format(args.arch))
    model = resnet.resnet50(zero_init_residual=args.zero_init_residual)

    # for ipex path, always convert model to channels_last for bf16, fp32.
    if args.ipex:
        model = model.to(memory_format=torch.channels_last)

    # Loss function (criterion)
    if 0 < args.label_smoothing < 1.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.base_op.lower() == "lars":
        print("Creating LARS optimizer")
        optimizer = create_optimizer_lars(model=model, lr=args.base_lr, epsilon=args.epsilon,
                                          momentum=args.momentum, weight_decay=args.weight_decay,
                                          bn_bias_separately=args.bn_bias_separately)
    else:                 
        print("Creating SGD optimizer")                         
        optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.ipex:
        print("Using fused LARS step")
        optimizer = ipex.optim._optimizer_utils.optimizer_fusion(optimizer, master_weight_split=False)


    if args.distributed:
        
        dist.init_process_group(backend=args.dist_backend, 
                                init_method=args.dist_url,
                                world_size=args.world_size, 
                                rank=args.rank)

        args.batch_size = int( args.batch_size / args.world_size)
        print("Using local batch size: ", args.batch_size)    

        dist._DEFAULT_FIRST_BUCKET_BYTES = 0
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False, broadcast_buffers=False,
                                                                 gradient_as_bucket_view=True, bucket_cap_mb=50)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))


    mllogger.end(key=INIT_STOP, sync=True)    

    # Data loading code
    assert args.data != None, "please set dataset path"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # Train loader
    traindir = os.path.join(args.data, 'train')
    mllogger.start(key=RUN_START, sync=True)
    train_dataset = datasets.ImageFolder(traindir,
                                        transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # Set affinity for data loader workers
    if args.distributed:
        worker_affinity=get_affinity_list(dist.get_rank(), args.workers)
    else:
        worker_affinity=get_affinity_list(0, args.workers) 


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, worker_affinity=worker_affinity, 
        persistent_workers=True)

    # Validation loader
    valdir = os.path.join(args.data, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]))

    if args.distributed:
        val_sampler = DistributedValSampler(val_dataset)
    else:
        val_sampler = None 

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, worker_affinity=worker_affinity)

    mllogger.event(key=TRAIN_SAMPLES, value=len(train_dataset))
    mllogger.event(key=EVAL_SAMPLES, value=len(val_dataset))

    num_steps_per_epoch = len(train_loader)
    if args.base_op.lower() == "lars":
        lr_scheduler = MLPerfLRScheduler(optimizer, args.epochs, args.warmup_epochs,
                                                    num_steps_per_epoch, args.base_lr,
                                                    args.end_lr, args.poly_power)
    else:
        lr_scheduler=None

    status = ABORTED
    for epoch in range(args.start_epoch+1, args.epochs+1):

        if (epoch==1):
            block_start_epoch = epoch
            mllogger.start(key=BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": args.eval_offset})
        elif (epoch%args.eval_period==args.eval_offset%args.eval_period+1):
            block_start_epoch = epoch
            mllogger.start(key=BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": args.eval_period})
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.base_op.lower() == "sgd":    
            adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args)

        # Sync BN stats among ranks
        if args.distributed:
            world_size = float(dist.get_world_size())
            for name, buff in model.named_buffers(): 
                if ('running_mean' in name) or ('running_var' in name):
                    dist.all_reduce(buff, op=dist.ReduceOp.SUM) 
                    buff /= world_size

        # Evaluate on validation set
        acc1 = 0
        if (epoch%args.eval_period==args.eval_offset%args.eval_period) and (epoch>=args.eval_offset):
            acc1 = validate(val_loader, model, criterion, epoch, args)
            mllogger.end(key=BLOCK_STOP, metadata={"first_epoch_num": block_start_epoch})

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank == 0 and args.save_checkpoint:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

        if (best_acc1>args.target_acc) and (epoch>=args.target_epoch):
            status=SUCCESS
            break

    mllogger.end(key=RUN_STOP, metadata={"status":status}, sync=True)
    mllogger.event(key=STATUS, value=status)

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args):

    mllogger.start(key=EPOCH_START, metadata={"epoch_num": epoch}, sync=True)

    batch_time = AverageMeter('Time', ':6.3f')      # Track total time = data-load + compute
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Train mode
    model.train()

    start = time.time()
    for i, (images, target) in enumerate(train_loader):

        if args.ipex:
            images = images.contiguous(memory_format=torch.channels_last)

        # Forward pass
        if args.ipex and args.bf16:
            with torch.cpu.amp.autocast():
                output = model(images)
            output = output.to(torch.float32)
        else:
            output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc, counts = accuracy(output, target, topk=(1, 5))
        acc1, acc5 = acc[0], acc[1]        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        if lr_scheduler: 
            lr_scheduler.step()
        start_opt = time.time()
        optimizer.step()

        # measure elapsed time and reset timer
        batch_time.update(time.time() - start)
        start = time.time()   

        if i % args.print_freq == 0:
            progress.display(i)

    perf = args.batch_size / (batch_time.avg)
    print("Training throughput: {:.3f} fps".format(perf))
    perf_tensor = torch.tensor(perf)
    dist.reduce(perf_tensor, 0, dist.ReduceOp.SUM)
    
    mllogger.event(key="throughput", value=perf_tensor.item(), metadata={"epoch_num":epoch}, sync=True)
    mllogger.end(key=EPOCH_STOP, metadata={"epoch_num": epoch}, sync=True)

'''
Distributed evaluation notes:

Default behavior of DDP is to broadcast named buffers from rank 0 to all ranks 
before FWD pass of each iteration. This ensures that BN stats are identical for all 
ranks at evaluation time, and top1 count can be aggregated over all ranks for accuracy
calculation. To remove broadcast overhead, we synchronize BN stats only before validation.

We also use a custom sampler for validation to avoid padding of val dataset with extra samples
in the distributed case. 
'''
def validate(val_loader, model, criterion, epoch, args):
    mllogger.start(key=EVAL_START, metadata={"epoch_num": epoch}, sync=True)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    top1_count=0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            start = time.time()

            if args.ipex:
                images = images.contiguous(memory_format=torch.channels_last)
            
            if args.ipex and args.bf16:
                images = images.to(torch.bfloat16)
                with torch.cpu.amp.autocast():
                    output = model(images)
                output = output.to(torch.float32) 
            else: 
                output = model(images) 

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, counts = accuracy(output, target, topk=(1, 5))
            acc1, acc5 = acc[0], acc[1]
            count1, _ = counts[0], counts[1]
            top1_count += count1.tolist()[0]            
            batch_time.update(time.time() - start)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

    top1_count = torch.tensor(top1_count)
    if args.distributed:
        dist.barrier()
        dist.all_reduce(top1_count, op=dist.ReduceOp.SUM) 
    top1_accuracy =  top1_count.tolist()*(1.0/50000)  
    print("Validation top1 accuracy after epoch {epoch}: {top1} ".format(epoch=epoch, top1=top1_accuracy))

    mllogger.event(key=EVAL_ACCURACY, value=top1_accuracy, metadata={"epoch_num": epoch})
    mllogger.end(key=EVAL_STOP, metadata={"epoch_num": epoch}, sync=True)

    return top1_accuracy


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        counts = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).int().sum(0, keepdim=True)
            res.append(correct_k.float().mul_(100.0 / batch_size))
            counts.append(correct_k)
        return res, counts


if __name__ == '__main__':
    main()
