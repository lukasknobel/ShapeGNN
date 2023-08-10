import enum
import os

from src.data_handling.CIFARCpp import CIFARCpp


from src.data_handling.ImageNetMapping import create_imagenet_mapping


class SupportedDatasets(enum.Enum):
    CIFAR10 = 1
    IMAGENET = 2


class Splits(enum.Enum):
    train = 1
    test = 2


def get_dataset_dir(root, dataset_type: SupportedDatasets, sub_dir=None):
    if sub_dir is None:
        return os.path.join(root, dataset_type.name.lower())
    else:
        return os.path.join(root, dataset_type.name.lower(), sub_dir)

def is_preprocessed_data_available(root, processed_dir, dataset_type: SupportedDatasets, split: Splits):
    if dataset_type is not SupportedDatasets.CIFAR10 and dataset_type is not SupportedDatasets.IMAGENET:
        raise NotImplementedError(f'Dataset {dataset_type} not supported')
    print("root and root processed dir: ", root, processed_dir)
    if not os.path.isdir(root):
        print(f'Creating data directory {root}')
        os.makedirs(root)
    if not os.path.isdir(processed_dir):
        print(f'Creating processed_dir directory {processed_dir}')
        os.makedirs(processed_dir)
        return False
    else:
        current_processed_location = os.path.join(processed_dir,split.name+'.npy')
        if not os.path.exists(current_processed_location):
            return False
    return True


def make_dataset_available_for_preprocessing(root, data_dir, processed_dir, dataset_type: SupportedDatasets, split: Splits, download=True):
    if dataset_type is SupportedDatasets.CIFAR10:
        if split is Splits.train:
            split_bool = True
        elif split is Splits.test:
            split_bool = False
        else:
            raise ValueError(f'Split {split} not supported for dataset {dataset_type}')
        CIFARCpp(root, split_bool, download=download)
    elif dataset_type is SupportedDatasets.IMAGENET:
        if download:
            print(f'Ignoring download argument for {dataset_type}')
        create_imagenet_mapping(data_dir, processed_dir)
