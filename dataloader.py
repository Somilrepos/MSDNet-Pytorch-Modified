import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    eval_only = args.evalmode is not None and not args.use_valid and args.eval_split != 'train'
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = None
        if not eval_only:
            train_set = datasets.CIFAR10(args.data_root, train=True,
                                         transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize
                                         ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
        if args.verbose:
            print('Dataset: CIFAR10')
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = None
        if not eval_only:
            train_set = datasets.CIFAR100(args.data_root, train=True,
                                          transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize
                                          ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
        if args.verbose:
            print('Dataset: CIFAR100')
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = None
        if not eval_only:
            train_set = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))
        need_val = args.use_valid or args.evalmode == 'dynamic' or args.eval_split in ['val', 'test']
        val_set = None
        if need_val:
            val_set = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]))
        if args.verbose:
            print('Dataset: ImageNet')
            print('Train dir: {}'.format(traindir))
            print('Val dir: {}'.format(valdir))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=True)
            if args.verbose:
                print('Train loader: {} samples'.format(len(train_set_index[:-num_sample_valid])))
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=True)
            if args.verbose:
                print('Val loader (subset): {} samples'.format(len(train_set_index[-num_sample_valid:])))
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            if args.verbose:
                print('Test loader: {} samples'.format(len(val_set)))
    else:
        if 'train' in args.splits and train_set is not None:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            if args.verbose:
                print('Train loader: {} samples'.format(len(train_set)))
        if ('val' in args.splits or 'test' in args.splits) and val_set is not None:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader
            if args.verbose:
                print('Val/Test loader: {} samples'.format(len(val_set)))

    return train_loader, val_loader, test_loader
