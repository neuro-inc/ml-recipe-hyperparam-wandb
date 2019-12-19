from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import catalyst.dl.callbacks as clb
import torchvision.transforms as t
from catalyst.dl.runner import SupervisedWandbRunner
from torch import Tensor
from torch import device as tdevice
from torch import nn
from torch.cuda import is_available
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

ROOT = Path(__file__).absolute().parent.parent
DATA_PATH = ROOT / 'data'
LOG_DIR = ROOT / 'results'


def get_imagenet_transforms() -> t.Compose:
    std = (0.229, 0.224, 0.225)
    mean = (0.485, 0.456, 0.406)
    transforms = t.Compose([t.ToTensor(), t.Normalize(mean=mean, std=std)])
    return transforms


class CifarWrapper:

    def __init__(self, **args: Any):
        self._dataset = CIFAR10(**args)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, label = self._dataset[idx]
        return {'features': img, 'targets': label}

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def classes(self) -> List[str]:
        return self._dataset.classes


class CifarResnet18(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()

        self._model = resnet18(pretrained=True)
        self._model.avgpool = nn.AdaptiveAvgPool2d(1)

        hide_dim = self._model.fc.in_features
        self._model.fc = nn.Linear(in_features=hide_dim,
                                   out_features=n_classes)

    def forward(self, img_batch: Tensor) -> Dict[str, Tensor]:
        return {'logits': self._model(img_batch)}


def main(args: Namespace) -> None:
    cur_time = str(datetime.now()).replace(' ', '_')
    logdir = args.logdir / f'logs_{cur_time}'
    logdir.mkdir(exist_ok=True, parents=True)

    cifar_args = {'transform': get_imagenet_transforms(),
                  'root': DATA_PATH, 'download': True}

    loader_args = {'batch_size': args.batch_size, 'num_workers': 4}

    train_loader = DataLoader(CifarWrapper(train=True, **cifar_args),
                              shuffle=True, **loader_args)

    test_loader = DataLoader(CifarWrapper(train=False, **cifar_args),
                             shuffle=False, **loader_args)

    n_classes = len(train_loader.dataset.classes)

    model = CifarResnet18(n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(params=model.parameters(), lr=args.lr)

    runner = SupervisedWandbRunner(
        input_key='features', input_target_key='targets', output_key=None,
        device=tdevice('cuda:0') if is_available() else tdevice('cpu')
    )

    callbacks = [
        clb.AccuracyCallback(prefix='accuracy', accuracy_args=[1],
                             output_key='logits', input_key='targets',
                             threshold=.5, num_classes=n_classes,
                             activation='Softmax'),
        clb.EarlyStoppingCallback(patience=3, minimize=False,
                                  min_delta=0.01, metric='accuracy01')
    ]
    runner.train(model=model, criterion=criterion, optimizer=optimizer,
                 loaders=OrderedDict([('train', train_loader), ('valid', test_loader)]),
                 logdir=logdir, num_epochs=args.n_epoch, verbose=True,
                 main_metric='accuracy01', valid_loader='valid', minimize_metric=False,
                 monitoring_params={'project': 'hyper_search'}, callbacks=callbacks
                 )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=Path, default=LOG_DIR)

    # args which can be optimized
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=256)

    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
