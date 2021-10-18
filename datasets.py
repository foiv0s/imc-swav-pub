import os
from sys import platform
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import socket
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import pickle
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

training_datasets = ['C10', 'C20', 'C100', 'STL10', 'TINY']


def get_encoder_size(dataset_name):
    if dataset_name in training_datasets[:3]:
        return 32
    if dataset_name == training_datasets[-2]:
        return 96
    if dataset_name == training_datasets[-1]:
        return 64
    raise RuntimeError("Error get encoder size, unknown setup size: {}".format(dataset_name))


def get_dataset(dataset_name):
    dataset_name = dataset_name.upper()
    if dataset_name in training_datasets:
        return dataset_name
    raise KeyError("Unknown dataset '" + dataset_name + "'. Must be one of "
                   + ', '.join([name for name in training_datasets]))


class Transforms:

    def __init__(self, nmb_crops, size_crops, min_scale_crops, max_scale_crops, mu, std):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        flip = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=mu, std=std)

        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i]))
            trans.extend([transforms.Compose([
                flip, randomresizedcrop, col_jitter, rnd_gray, transforms.ToTensor(), normalize])] * nmb_crops[i])
        self.train_transform = trans

        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __call__(self, inp):
        multi_crops = list(map(lambda trans: trans(inp), self.train_transform))
        return multi_crops, self.test_transform(inp)


def build_dataset(dataset, batch_size, nmb_workers, nmb_crops, size_crops, min_scale_crops, max_scale_crops, path):
    if dataset == training_datasets[0]:
        num_classes = 10
        mu = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_transform = Transforms(nmb_crops, size_crops, min_scale_crops, max_scale_crops, mu, std)
        test_transform = train_transform.test_transform
        train_dataset = CIFAR10(root=path, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root=path, train=False, transform=test_transform, download=True)

    elif dataset in training_datasets[1:3]:
        num_classes = 20 if dataset == training_datasets[1] else 100
        coarse = True if dataset == training_datasets[1] else False
        mu = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        train_transform = Transforms(nmb_crops, size_crops, min_scale_crops, max_scale_crops, mu, std)
        test_transform = train_transform.test_transform
        train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True, c100_coarse=coarse)
        test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=True, c100_coarse=coarse)

    elif dataset == training_datasets[-2]:
        num_classes = 10
        mu = [0.43, 0.42, 0.39]
        std = [0.27, 0.26, 0.27]
        train_transform = Transforms(nmb_crops, size_crops, min_scale_crops, max_scale_crops, mu, std)
        test_transform = train_transform.test_transform
        train_dataset = STL10(root=path, split='train+unlabeled', transform=train_transform, download=True)
        test_dataset = STL10(root=path, split='test', transform=test_transform, download=True)
    elif dataset == training_datasets[-1]:
        num_classes = 200
        mu = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = Transforms(nmb_crops, size_crops, min_scale_crops, max_scale_crops, mu, std)
        test_transform = train_transform.test_transform
        train_dataset = TinyImageNetDataset(path, transform=train_transform, download=False, preload=True)
        test_dataset = TinyImageNetDataset(path, mode='val', transform=test_transform, download=False,
                                           preload=False)
    else:
        raise RuntimeError("Error not supported dataset {}".format(dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, drop_last=True, num_workers=nmb_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                              pin_memory=True, drop_last=False, num_workers=nmb_workers)

    return train_loader, test_loader, num_classes


'''
Overwritting Pytorch methods of CIFAR-10 and CIFAR-100 for being able to provide the coarse labels (CIFAR-20)
Additionally, all below vision datasets return the index of the iterating instance for reference only
'''


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, c100_coarse=True):

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    if c100_coarse is True:
                        self.targets.extend(entry['coarse_labels'])
                        self.meta['key'] = self.meta['key2']
                    else:
                        self.targets.extend(entry['fine_labels'])
                        self.meta['key'] = self.meta['key1']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key1': 'fine_label_names',
        'key2': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class STL10(datasets.STL10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # img = img.crop((8, 8, 88, 88))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img):
    if len(img.getbands()) == 1:  # third axis is the channels
        img = np.expand_dims(np.array(img), -1)
        img = np.tile(img, (1, 1, 3))
        img = Image.fromarray(img)
    return img


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = {}  # np.zeros((self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                # img = imageio.imread(s[0])
                img_ = Image.open(s[0])
                img = img_.copy()
                img = _add_channels(img)
                img_.close()
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            target = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            # img = imageio.imread(s[0])
            img = Image.open(s[0])
            img = _add_channels(img)
            target = None if self.mode == 'test' else s[self.label_idx]
        # img = img.crop((4, 4, 60, 60))

        # to return a PIL Image
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        return img, target, idx


dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""
