import os
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torchvision.transforms.functional as TF
import cv2
from prepgress import Blurfilter, GaussianNoise, RandomAffine, RandomResizedCrop, RandomRotation, \
    Stage2Resize, Stage2_ToPILImage, Stage2ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset



np.set_printoptions(threshold=np.inf)


class HelenDataset(Dataset):
    # HelenDataset

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.root_dir, 'images',
                                img_name + '.jpg')
        labels_path = [os.path.join(self.root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]

        image = io.imread(img_path)
        image = np.array(image)
        labels = [io.imread(labels_path[i]) for i in range(11)]
        labels = np.array(labels)
        # bg = labels[0] + labels[1] + labels[10]
        bg = 255 - labels[2:10].sum(0)
        labels = np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0)
        sample = {'image': image, 'labels': labels, 'orig': image, 'orig_label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PartsDataset(Dataset):
    #     # HelenDataset
    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        self.names = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']
        self.label_id = {'eyebrow1': [2],
                         'eyebrow2': [3],
                         'eye1': [4],
                         'eye2': [5],
                         'nose': [6],
                         'mouth': [7, 8, 9]
                         }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        part_path = [os.path.join(self.root_dir, '%s' % x, 'images',
                                  img_name + '.jpg')
                     for x in self.names]
        labels_path = {x: [os.path.join(self.root_dir, '%s' % x,
                                        'labels', img_name,
                                        img_name + "_lbl%.2d.png" % i)
                           for i in self.label_id[x]]
                       for x in self.names}

        parts_image = [io.imread(part_path[i])
                       for i in range(6)]

        labels = {x: np.array([io.imread(labels_path[x][i])
                               for i in range(len(self.label_id[x]))
                               ])
                  for x in self.names
                  }

        for x in self.names:
            bg = 255 - np.sum(labels[x], axis=0, keepdims=True)  # [1, 64, 64]
            labels[x] = np.uint8(np.concatenate([bg, labels[x]], axis=0))  # [L + 1, 64, 64]

        # labels = {'eyebrow1':,
        #           'eyebrow2':,
        #           'eye1':,
        #           'eye2':,
        #           'nose':,
        #           'mouth':}

        sample = {'image': parts_image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
            # 去掉数据增广后的背景黑边
            img, new_label = sample['image'], sample['labels']
            new_label_fg = {x: torch.sum(new_label[x][1:], dim=0, keepdim=True)  # 1 x 64 x 64
                            for x in ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']}
            for x in ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']:
                new_label[x][0] = 1 - new_label_fg[x]
            sample = {'image': img, 'labels': new_label}
        return sample


class Stage2Augmentation(object):
    def __init__(self, dataset, txt_file, root_dir, resize):
        self.augmentation_name = ['origin', 'choice1', 'choice2', 'choice3', 'choice4']
        self.randomchoice = None
        self.transforms = None
        self.transforms_list = None
        self.dataset = dataset
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.resize = resize
        self.set_choice()
        self.set_transformers()
        self.set_transforms_list()

    def set_choice(self):
        choice = {
            # random_choice 1:  Blur, rotaion, Blur + rotation + scale_translate (random_order)
            self.augmentation_name[1]: [Blurfilter(),
                                        RandomRotation(15),
                                        transforms.RandomOrder([Blurfilter(),
                                                                RandomRotation(15),
                                                                RandomAffine(degrees=0, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 2:  noise, crop, noise + crop + rotation_scale_translate (random_order)
            self.augmentation_name[2]: [GaussianNoise(),
                                        RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                        RandomAffine(degrees=15, translate=(0.01, 0.1), scale=(0.9, 1.1)),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 3:  noise + blur , noise + rotation ,noise + blur + rotation_scale_translate
            self.augmentation_name[3]: [transforms.RandomOrder([GaussianNoise(),
                                                                Blurfilter()
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                RandomRotation(15)
                                                                ]
                                                               ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                Blurfilter(),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ],
            # random_choice 4:  noise + crop , blur + crop ,noise + blur + crop + rotation_scale_translate
            self.augmentation_name[4]: [transforms.RandomOrder([GaussianNoise(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                                ]
                                                               ),
                                        transforms.Compose([Blurfilter(),
                                                            RandomResizedCrop((64, 64), scale=(0.9, 1.1))
                                                            ]
                                                           ),
                                        transforms.RandomOrder([GaussianNoise(),
                                                                Blurfilter(),
                                                                RandomResizedCrop((64, 64), scale=(0.9, 1.1)),
                                                                RandomAffine(degrees=15, translate=(0.01, 0.1),
                                                                             scale=(0.9, 1.1))
                                                                ]
                                                               )
                                        ]
        }
        self.randomchoice = choice

    def set_resize(self, resize):
        self.resize = resize

    def set_transformers(self):
        self.transforms = {
            self.augmentation_name[0]: transforms.Compose([
                Stage2Resize(self.resize),
                Stage2_ToPILImage(),
                Stage2ToTensor()
            ]),
            self.augmentation_name[1]: transforms.Compose([
                Stage2Resize(self.resize),
                Stage2_ToPILImage(),
                # Choose from tranforms_list randomly
                transforms.RandomChoice(self.randomchoice['choice1']),
                Stage2ToTensor()
            ]),
            self.augmentation_name[2]: transforms.Compose([
                Stage2Resize(self.resize),
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice2']),
                Stage2ToTensor()
            ]),
            self.augmentation_name[3]: transforms.Compose([
                Stage2Resize(self.resize),
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice3']),
                Stage2ToTensor()
            ]),
            self.augmentation_name[4]: transforms.Compose([
                Stage2Resize(self.resize),
                Stage2_ToPILImage(),
                transforms.RandomChoice(self.randomchoice['choice4']),
                Stage2ToTensor()
            ])
        }

    def set_transforms_list(self):
        self.transforms_list = {
            'train':
                self.transforms,
            'val':
                self.transforms['origin']
        }

    def get_dataset(self):
        datasets = {'train': [self.dataset(txt_file=self.txt_file['train'],
                                           root_dir=self.root_dir,
                                           transform=self.transforms_list['train'][r]
                                           )
                              for r in self.augmentation_name],
                    'val': self.dataset(txt_file=self.txt_file['val'],
                                        root_dir=self.root_dir,
                                        transform=self.transforms_list['val']
                                        )
                    }
        enhaced_datasets = {'train': ConcatDataset(datasets['train']),
                            'val': datasets['val']
                            }

        return enhaced_datasets
