# The code is modified from https://github.com/pytorch/examples/blob/main/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
import logging
from enum import Enum
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_NORMALIZATION_OPS"] = "1"

import oneflow
import oneflow.backends.cudnn as cudnn
import oneflow.distributed as dist
import oneflow.multiprocessing as mp
import oneflow.nn as nn
import oneflow.nn.parallel
import oneflow.optim
import oneflow.utils.data
import oneflow.utils.data.distributed
import flowvision.datasets as datasets
import flowvision.models as models
import flowvision.transforms as transforms
from oneflow.optim.lr_scheduler import StepLR
from oneflow.utils.data import Subset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Pyoneflow ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='/ssd/dataset/ImageNet/extract',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=192, type=int,
                    metavar='N',
                    help='mini-batch size (default: 192), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use Pyoneflow for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--benchmark', action='store_true',
                    help="benchmark by inferencing backbone with constant input")
parser.add_argument('--channels-last', action='store_true',
                    help="Use NHWC memory format instead of NCHW")
parser.add_argument('--logfile', default=None, help="the filepath to save log")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        oneflow.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if oneflow.cuda.is_available():
        ngpus_per_node = oneflow.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use oneflow.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        main_worker(args.gpu, ngpus_per_node, args)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # args.rank = args.rank * ngpus_per_node + gpu
            args.rank = oneflow.env.get_rank()
        # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                         world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if args.benchmark:
            args.total_flops, args.total_params = get_model_FLOPs(model, (1, 3, 224, 224))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        if args.benchmark:
            args.total_flops, args.total_params = get_model_FLOPs(model, (1, 3, 224, 224))

    if not oneflow.cuda.is_available() and not oneflow.backends.mps.is_available():
        pass
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if oneflow.cuda.is_available():
            if args.gpu is not None:
                oneflow.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = oneflow.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = oneflow.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and oneflow.cuda.is_available():
        oneflow.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif oneflow.backends.mps.is_available():
        device = oneflow.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = oneflow.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = oneflow.nn.DataParallel(model).cuda()

    if oneflow.cuda.is_available():
        if args.gpu:
            device = oneflow.device('cuda:{}'.format(args.gpu))
        else:
            device = oneflow.device("cuda")
    elif oneflow.backends.mps.is_available():
        device = oneflow.device("mps")
    else:
        # device = oneflow.device("cpu")
        device = oneflow.device("mlu")
    
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=oneflow.channels_last)
    
    if args.distributed:
        model = oneflow.nn.parallel.DistributedDataParallel(model)
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = oneflow.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = oneflow.load(args.resume)
            elif oneflow.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = oneflow.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.logger = get_logger(args.logfile, args.rank)
    args.logger.info("options: ")
    args.logger.info(vars(args))

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = oneflow.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = oneflow.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = oneflow.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = oneflow.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    if args.benchmark:
        
        benchmark(val_loader, model, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    samples = AverageMeter('Throughput', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, samples],
        args.logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device)
        if args.channels_last:
            images = images.to(memory_format=oneflow.channels_last)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time_value = time.time() - end
        batch_time.update(batch_time_value)
        samples.update(args.batch_size / batch_time_value)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, device, args):

    def run_validate(loader, base_progress=0):
        with oneflow.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and oneflow.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if oneflow.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if oneflow.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                images = images.to(device)
                if args.channels_last:
                    images = images.to(memory_format=oneflow.channels_last)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time_value = time.time() - end
                batch_time.update(batch_time_value)
                samples.update(args.batch_size / batch_time_value)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    samples = AverageMeter('Throughput', ':6.2f')

    progress_list = [batch_time, losses, top1, top5, samples]
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5, samples],
        args.logger,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()


    args.world_size = oneflow.env.get_world_size()
    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = oneflow.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

def benchmark(val_loader, model, device, args):
    images = oneflow.randn(args.batch_size, 3, 224, 224).to(device)
    if args.channels_last:
        images = images.to(memory_format=oneflow.channels_last)
    iter_count = 100

    def run_benchmark(loader, base_progress=0):
        class ResNet50Graph(nn.Graph):
            def __init__(self):
                super().__init__()
                self.model = model

            def build(self, input):
                return self.model(input)

        resnet50_graph = ResNet50Graph()
    
        with oneflow.no_grad():
            # warmup 5 iters
            for i in range(5):
                output = resnet50_graph(images)
            output.numpy()

            end = time.time()
            for i in range(iter_count):
                output = resnet50_graph(images)

        oneflow._oneflow_internal.eager.Sync()
        batch_time_value = time.time() - end
        batch_time.update(batch_time_value)
        samples.update(iter_count * args.batch_size / batch_time_value)
        flops.update(iter_count * args.batch_size * args.total_flops / 1e12 / batch_time_value)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    samples = AverageMeter('Throughput', ':6.2f')
    flops = AverageMeter('FLOPS(T/s)', ':6.2f')

    progress = ProgressMeter(
        iter_count,
        [batch_time, samples, flops],
        args.logger,
        prefix='Benchmark: ')

    # switch to evaluate mode
    model.eval()
    if args.channels_last:
        model.to(memory_format=oneflow.channels_last)

    run_benchmark(val_loader)
    progress.display_summary()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    oneflow.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if oneflow.cuda.is_available():
            device = oneflow.device("cuda")
        elif oneflow.backends.mps.is_available():
            device = oneflow.device("mps")
        else:
            # device = oneflow.device("cpu")
            device = oneflow.device("mlu")
        total = oneflow.tensor([self.sum, self.count], dtype=oneflow.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        self.logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with oneflow.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_model_FLOPs(model, input_size):
    try:
        from flowflops import get_model_complexity_info
        from flowflops.utils import flops_to_string, params_to_string
        from flowflops.flow_engine import reset_flops_count
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install `flowflops` by `pip install flowflops -i https://pypi.tuna.tsinghua.edu.cn/simple`")

    total_flops, total_params = get_model_complexity_info(
        model, input_size,
        as_strings=False,
        print_per_layer_stat=False,
        mode="eager",
    )

    # remove hooks in module
    reset_flops_count(model)
    return total_flops * 2, total_params

def get_logger(filename, rank, verbosity=1, name=None):
    if rank > 0:
        return logging.getLogger(None)

    if filename is None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"./imagenet_log_{timestr}.txt"

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    return logger

if __name__ == '__main__':
    main()
