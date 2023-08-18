from torchvision.transforms import Compose, ToTensor, Resize
from dataset import DatasetFromFolderEval, DatasetFromFolder


def transform():
    return Compose([
        ToTensor(),
        Resize((256, 256))
    ])


def get_training_set(data_dir, data_train_RMT, label_dir, data_augmentation):
    return DatasetFromFolder(data_dir, data_train_RMT, label_dir, data_augmentation, transform=transform())


def get_eval_set(data_dir, data_test_RMT,  label_dir):
    return DatasetFromFolderEval(data_dir, data_test_RMT, label_dir, transform=transform())



