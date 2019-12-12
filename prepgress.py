import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2 as cv
import numpy as np
import random
from skimage.util import random_noise
from PIL import ImageFilter, Image
import cv2


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']

        resized_image = TF.resize(image, self.size, Image.ANTIALIAS)
        resized_labels = [TF.resize(labels[r], self.size, Image.ANTIALIAS)
                          for r in range(len(labels))
                          ]
        resized_olabels = [TF.resize(orig_label[r], (512, 512), Image.ANTIALIAS)
                           for r in range(len(orig_label))
                           ]

        sample = {'image': resized_image,
                  'labels': resized_labels,
                  'orig': cv2.resize(sample['orig'], (512, 512)),
                  'orig_label': resized_olabels
                  }

        return sample


class ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']
        labels = np.uint8(labels)
        orig_label = np.uint8(orig_label)
        image = TF.to_pil_image(image)

        labels = [TF.to_pil_image(labels[i])
                  for i in range(labels.shape[0])]

        orig_label = [TF.to_pil_image(orig_label[i])
                      for i in range(orig_label.shape[0])]

        return {'image': image,
                'labels': labels,
                'orig': sample['orig'],
                'orig_label': orig_label
                }


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image, labels = sample['image'], sample['labels']
        orig_label = sample['orig_label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # Don't need to swap label because
        # label image: 9 X H X W

        labels = [TF.to_tensor(labels[r])
                  for r in range(len(labels))
                  ]
        labels = torch.cat(labels, dim=0).float()
        labels = labels.argmax(dim=0).float()  # Shape(H, W)

        orig_label = [TF.to_tensor(orig_label[r])
                      for r in range(len(orig_label))
                      ]
        orig_label = torch.cat(orig_label, dim=0).float()
        orig_label = orig_label.argmax(dim=0).float()  # Shape(H, W)

        return {'image': TF.to_tensor(image),
                'labels': labels,
                'orig': TF.to_tensor(sample['orig']),
                'orig_label': orig_label
                }
