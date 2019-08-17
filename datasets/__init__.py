from datasets.seafood_det_train import SeafoodDETTrain
from datasets.seafood_det_test import SeafoodDETTest
from torch.utils.data import DataLoader
from .dataloader import Dataloader as _Dataloader


def make_train_dataloader(cfg):
    train_dataset = SeafoodDETTrain(root_dir=cfg.data_root, split='train', transforms=cfg.Train.transforms)

    train_loader = _Dataloader(DataLoader(train_dataset,
                                          batch_size=cfg.Train.batch_size, num_workers=cfg.Train.num_workers,
                                          sampler=cfg.Train.sampler(train_dataset) if cfg.Train.sampler else None,
                                          pin_memory=True, collate_fn=SeafoodDETTrain.collate_fn,
                                          shuffle=True if cfg.Train.sampler is None else False))
    return train_loader


def make_val_dataloader(cfg):
    test_dataset = SeafoodDETTest(root_dir=cfg.data_root, split='val', mean=cfg.mean, std=cfg.std)

    test_loader = DataLoader(test_dataset,
                             batch_size=1, num_workers=4,
                             sampler=None,
                             pin_memory=True,
                             shuffle=False)
    return test_loader


def make_test_dataloader(cfg):
    test_dataset = SeafoodDETTest(root_dir=cfg.data_root, split='test', mean=cfg.mean, std=cfg.std)

    test_loader = DataLoader(test_dataset,
                             batch_size=1, num_workers=4,
                             sampler=None,
                             pin_memory=True,
                             shuffle=False)
    return test_loader
