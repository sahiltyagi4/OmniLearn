from omnilearn_pytorch.helper import miscellaneous as misc

from torch import nn, optim
import torchvision.models as models
from transformers import GPT2LMHeadModel, AdamW

def get_model(model_name, determinism, args):

    if model_name == 'resnet50':
        model_obj = ResNet50Object(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed,
                                   gamma=args.gamma, determinism=determinism)
    elif model_name == 'alexnet':
        model_obj = AlexNetObject(lr=args.lr, seed=args.seed, gamma=args.gamma, momentum=args.momentum,
                                  weightdecay=args.weight_decay, determinism=determinism)

    elif model_name == 'vgg11':
        model_obj = VGG11Object(lr=args.lr, momentum=args.momentum, seed=args.seed, weight_decay=args.weight_decay,
                                gamma=args.gamma, determinism=determinism)

    elif model_name == 'resnet18':
        model_obj = ResNet18Object(lr=args.lr, momentum=args.momentum, seed=args.seed, weight_decay=args.weight_decay,
                                   gamma=args.gamma, determinism=determinism)
    elif model_name == 'gpt2':
        model_obj = GPT2ModelObject(model_name=model_name, lr=args.lr, seed=args.seed, determinism=determinism)
    else:
        raise ValueError('Invalid model name')

    return model_obj


class ResNet18Object(object):

    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.resnet18(progress=True, pretrained=False)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                               weight_decay=self.weightdecay)
        self.milestones = [15, 30, 45]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=self.milestones,
                                                           gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        else:
            raise TypeError

    def get_milestones(self):
        return self.milestones

    def get_lrgamma(self):
        return self.gamma


class ResNet50Object(object):

    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.resnet50(progress=True, pretrained=False)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weightdecay)
        self.milestones = [100, 150, 200]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=self.milestones,
                                                           gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        else:
            raise TypeError

    def get_milestones(self):
        return self.milestones

    def get_lrgamma(self):
        return self.gamma


class VGG11Object(object):

    def __init__(self, lr, momentum, seed, weight_decay, gamma, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.vgg11(pretrained=False, progress=True)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                               weight_decay=self.weight_decay)
        self.milestones = [15, 25, 35]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.milestones,
                                                           gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        else:
            raise TypeError

    def get_milestones(self):
        return self.milestones

    def get_lrgamma(self):
        return self.gamma


class AlexNetObject(object):
    def __init__(self, lr, gamma, seed, momentum, weightdecay, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.gamma = gamma
        self.momentum = momentum
        self.weightdecay = weightdecay
        self.loss = nn.CrossEntropyLoss()
        self.model = models.alexnet(progress=True, pretrained=False)
        self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum,
                             weight_decay=self.weightdecay)
        self.milestones = [80, 120, 160]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=self.milestones,
                                                           gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        else:
            raise TypeError

    def get_milestones(self):
        return self.milestones

    def get_lrgamma(self):
        return self.gamma


class GPT2ModelObject():
    def __init__(self, model_name, lr, seed, determinism):
        misc.set_seed(seed, determinism)
        self.lr = lr
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.opt = AdamW(self.model.parameters(), lr=self.lr)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt