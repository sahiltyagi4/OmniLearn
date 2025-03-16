import logging
import os.path
import random
import tarfile

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url

import miscellaneous as misc

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class BSPDataPartitioner(object):
    def __init__(self, data, world_size):
        self.data = data
        self.partitions = []
        # partition data equally among the trainers
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        partitions = [1 / (world_size) for _ in range(0, world_size)]
        print(f"partitions are {partitions}")

        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class ASPDataPartitioner(object):
    """Unlike bsp data partitioner which evenly distributes data among all workers, asp partitioner keeps all
    data on each worker and shuffles it in a different order"""
    def __init__(self, data, rank):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.seed(rank)
        random.shuffle(indexes)
        logging.info(f'ASP data partitioner for worker-rank {rank} top-10 ixs are {indexes[0:10]}')
        self.partitions.append(indexes)

    def use(self, partition_ix):
        return Partition(self.data, self.partitions[partition_ix])


class TrainingTestingDataset(object):
    def __init__(self, bsz, dataset_name, args, fetchtestdata=True):
        if dataset_name == 'cifar10':
            self.train_size = 50000
            self.trainloader = cifar10Train(log_dir=args.data_dir, world_size=args.world_size, trainer_rank=args.rank,
                                            train_bsz=bsz, seed=args.seed, bsp=args.bsp)
            if fetchtestdata:
                self.testloader = cifar10Test(log_dir=args.data_dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'cifar100':
            self.train_size = 50000
            self.trainloader = cifar100Train(log_dir=args.data_dir, world_size=args.world_size, trainer_rank=args.rank,
                                             train_bsz=bsz, seed=args.seed, bsp=args.bsp)
            if fetchtestdata:
                self.testloader = cifar100Test(log_dir=args.data_dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'food101':
            self.train_size = 75750
            self.trainloader = food101Train(log_dir=args.data_dir, world_size=args.world_size,
                                               trainer_rank=args.rank, train_bsz=bsz, seed=args.seed,
                                               bsp=args.bsp)
            if fetchtestdata:
                self.testloader = food101Test(log_dir=args.data_dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'caltech256':
            self.train_size = 30000
            self.trainloader = caltechTrain(log_dir=args.data_dir, bsz=bsz, seed=args.seed,
                                               world_size=args.world_size, trainer_rank=args.rank, bsp=args.bsp)
            if fetchtestdata:
                self.testloader = caltechTest(log_dir=args.data_dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'places365':
            self.train_size = 1.8e6
            self.trainloader = places365Train(log_dir=args.data_dir, world_size=args.world_size,
                                              trainer_rank=args.rank, train_bsz=bsz, seed=args.seed, bsp=True)
            if fetchtestdata:
                self.testloader = places365Test(log_dir=args.data_dir, test_bsz=args.test_bsz, seed=args.seed)

        else:
            raise ValueError('not a valid dataset name')

    def getTrainloader(self):
        return self.trainloader

    def getTestloader(self):
        return self.testloader

    def getTrainsize(self):
        return self.train_size


def cifar10Train(log_dir, world_size, trainer_rank, train_bsz, seed, bsp=True):
    print(f"going to use partition ix {trainer_rank}")
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    trainset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=True, download=True, transform=transform)
    if bsp:
        partition = BSPDataPartitioner(trainset, world_size)
        partition = partition.use(trainer_rank)
    else:
        partition = ASPDataPartitioner(trainset, trainer_rank)
        partition = partition.use(0)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g, num_workers=4)
    return trainloader

def cifar10Test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def cifar100Train(log_dir, world_size, trainer_rank, train_bsz, seed, bsp=True):
    print(f"going to use partition ix {trainer_rank}")
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.Resize(224), normalize])
    trainset = torchvision.datasets.CIFAR100(root=log_dir, train=True, download=True, transform=transform)

    if bsp:
        partition = BSPDataPartitioner(trainset, world_size)
        partition = partition.use(trainer_rank)
    else:
        partition = ASPDataPartitioner(trainset, trainer_rank)
        partition = partition.use(0)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g, num_workers=4)
    return trainloader

def cifar100Test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR100(root=log_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def food101Train(log_dir, world_size, trainer_rank, train_bsz, seed, bsp=True):
    print(f"going to use partition ix {trainer_rank}")
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.Food101(root=log_dir, split='train', transform=transform, download=True)
    if bsp:
        partition = BSPDataPartitioner(trainset, world_size)
        partition = partition.use(trainer_rank)
    else:
        partition = ASPDataPartitioner(trainset, trainer_rank)
        partition = partition.use(0)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g, num_workers=4)
    return trainloader

def food101Test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testset = torchvision.datasets.Food101(root=log_dir, split='test', transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def places365Train(log_dir, world_size, trainer_rank, train_bsz, seed, bsp=True):
    print(f"going to use partition ix {trainer_rank}")
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.Places365(root=log_dir, split='train', transform=transform, download=True)
    if bsp:
        partition = BSPDataPartitioner(trainset, world_size)
        partition = partition.use(trainer_rank)
    else:
        partition = ASPDataPartitioner(trainset, trainer_rank)
        partition = partition.use(0)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g, num_workers=4)
    return trainloader

def places365Test(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testset = torchvision.datasets.Places365(root=log_dir, split='test', transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def caltechTrain(log_dir, bsz, world_size, trainer_rank, seed, bsp=True):
    # download caltech-256 dataset if not present in data_dir
    if not os.path.isdir(os.path.join(log_dir, '256_ObjectCategories')):
        print("CalTech-256 dataset not present. Going to download it...")
        url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
        filename = "256_ObjectCategories.tar"
        md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"
        curr_dir = os.getcwd()
        download_url(url=url, filename=filename, md5=md5, root=log_dir)
        tar = tarfile.open(os.path.join(log_dir, filename), "r")
        os.chdir(log_dir)
        tar.extractall()
        tar.close()
        os.chdir(curr_dir)

    print(f"going to use partition ix {trainer_rank}")
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    dataset = datasets.ImageFolder(log_dir, transform=transform)
    if bsp:
        partition = BSPDataPartitioner(dataset, world_size)
        partition = partition.use(trainer_rank)
    else:
        partition = ASPDataPartitioner(dataset, trainer_rank)
        partition = partition.use(0)

    del dataset
    trainloader = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, worker_init_fn=misc.set_seed(seed),
                                              generator=g, num_workers=1)
    return trainloader

def caltechTest(log_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    dataset = datasets.ImageFolder(log_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)
    return testloader