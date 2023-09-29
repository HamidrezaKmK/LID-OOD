import numpy as np
from torch.utils.data import DataLoader

from .image import get_image_datasets
from .generated import get_generated_datasets, get_dgm_generated_datasets
from .supervised_dataset import SupervisedDataset

import typing as th
    
from .utils import SUPPORTED_IMAGE_DATASETS, SUPPORTED_GENERATED_DATASETS, OmitLabels, TrainerReadyDataset

def get_loader(dset, device, batch_size, drop_last, shuffle=True):
    return DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False,
    )
    
def get_loaders(
    dataset,
    device,
    data_root,
    train_batch_size,
    valid_batch_size,
    test_batch_size,
    make_valid_loader: bool = False,
    make_test_loader: bool = True,
    shuffle: bool = True,
    dgm_args: th.Optional[th.Dict[str, th.Any]] = None,
    train_ready: bool = False,
    unsupervised: bool = False,
):
    if dataset in SUPPORTED_IMAGE_DATASETS:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, make_valid_loader)
        
    elif dataset in SUPPORTED_GENERATED_DATASETS:
        train_dset, valid_dset, test_dset = get_generated_datasets(dataset)
        
    elif dataset == 'dgm-generated':
        train_dset, valid_dset, test_dset = get_dgm_generated_datasets(data_root, dgm_args)
        
    else:
        raise ValueError(f"Unknown dataset {dataset}, please check model_zoo/datasets/utils.py for supported ones!")
    
    # Use wrappers on the datasets if necessary
    # for example, without the wrapper for training, the dataset is not compatible
    # for training.
    # without the unsupervised wrapper, the data is not ready for unsupservised tasks
    if unsupervised:
        if train_ready:
            raise Exception("train_ready and unsupervised cannot be both true!")
        train_dset = OmitLabels(train_dset)
        valid_dset = OmitLabels(valid_dset)
        test_dset = OmitLabels(test_dset)
    if train_ready:
        train_dset = TrainerReadyDataset(train_dset)
        valid_dset = TrainerReadyDataset(valid_dset)
        test_dset = TrainerReadyDataset(test_dset)
    
    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=True, shuffle=shuffle)
    valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False, shuffle=False) if make_valid_loader else None
    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False, shuffle=False) if make_test_loader else None
    
    return train_loader, valid_loader, test_loader
    


def get_embedding_loader(embeddings, batch_size, drop_last, role):
    dataset = SupervisedDataset(
        name="embeddings",
        role=role,
        x=embeddings
    )
    return get_loader(dataset, embeddings.device, batch_size, drop_last)


def remove_drop_last(loader):
    dset = loader.dataset
    return get_loader(dset, dset.device, loader.batch_size, False)
