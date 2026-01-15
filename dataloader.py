import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class FlatImageNetVal(Dataset):
    def __init__(self, valdir, val_txt, synset_mapping, transform=None):
        self.valdir = valdir
        self.transform = transform
        self.loader = default_loader

        with open(synset_mapping, 'r') as f:
            synsets = [line.strip().split()[0] for line in f if line.strip()]
        self.class_to_idx = {syn: idx for idx, syn in enumerate(synsets)}

        self.samples = []
        with open(val_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_id, synset = parts[0], parts[1]
                if synset not in self.class_to_idx:
                    continue
                path = os.path.join(valdir, '{}.JPEG'.format(img_id))
                self.samples.append((path, self.class_to_idx[synset]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    eval_only = args.evalmode is not None and not args.use_valid
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
        val_txt = os.path.join(args.data_root, '..', 'ImageSets', 'CLS-LOC', 'val.txt')
        synset_mapping = os.path.join(args.data_root, '..', '..', 'LOC_synset_mapping.txt')
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
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        has_class_dirs = False
        if os.path.isdir(valdir):
            for entry in os.listdir(valdir):
                if os.path.isdir(os.path.join(valdir, entry)):
                    has_class_dirs = True
                    break
        if has_class_dirs:
            val_set = datasets.ImageFolder(valdir, val_transform)
        else:
            val_set = FlatImageNetVal(valdir, val_txt, synset_mapping, val_transform)
        if args.verbose:
            print('Dataset: ImageNet')
            print('Train dir: {}'.format(traindir))
            print('Val dir: {}'.format(valdir))
            print('Val layout: {}'.format('class-folders' if has_class_dirs else 'flat'))
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
        if 'val' in args.splits or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader
            if args.verbose:
                print('Val/Test loader: {} samples'.format(len(val_set)))

    return train_loader, val_loader, test_loader
