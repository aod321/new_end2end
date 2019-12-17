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

name_list = ['eyebrow1', 'eyebrow2', 'eye1', 'eye2', 'nose', 'mouth']

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


class Stage2Resize(transforms.Resize):
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
        resized_image = np.array([cv2.resize(image[i], self.size, interpolation=cv2.INTER_AREA)
                                  for i in range(len(image))])
        labels = {x: np.array([np.array(TF.resize(TF.to_pil_image(labels[x][r]), self.size, Image.ANTIALIAS))
                               for r in range(len(labels[x]))])
                  for x in name_list
                  }

        sample = {'image': resized_image,
                  'labels': labels
                  }

        return sample


class Stage2ToTensor(transforms.ToTensor):
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
        image = sample['image']
        labels = sample['labels']
        image = torch.stack([TF.to_tensor(image[i])
                             for i in range(len(image))])

        labels = {x: torch.cat([TF.to_tensor(labels[x][r])
                                for r in range(len(labels[x]))
                                ])
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }


class Stage2_ToPILImage(object):
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
        image = [TF.to_pil_image(image[i])
                 for i in range(len(image))]
        labels = {x: [TF.to_pil_image(labels[x][i])
                      for i in range(len(labels[x]))]
                  for x in name_list
                  }

        return {'image': image,
                'labels': labels
                }


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.

        Override the __call__ of transforms.RandomRotation

    """

    def __call__(self, sample):
        """
            sample (dict of PIL Image and label): Image to be rotated.

        Returns:
            Rotated sample: dict of Rotated image.
        """

        angle = self.get_params(self.degrees)

        img, labels = sample['image'], sample['labels']

        rotated_img = [TF.rotate(img[i], angle, self.resample, self.expand, self.center)
                       for i in range(len(img))]
        rotated_labels = {x: [TF.rotate(labels[x][r], angle, self.resample, self.expand, self.center)
                              for r in range(len(labels[x]))
                              ]
                          for x in name_list
                          }

        sample = {'image': rotated_img,
                  'labels': rotated_labels
                  }

        return sample


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.RandomResizedCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']

        croped_img = []
        for r in range(len(img)):
            i, j, h, w = self.get_params(img[r], self.scale, self.ratio)
            croped_img.append(TF.resized_crop(img[r], i, j, h, w, self.size, self.interpolation))


        croped_labels = {x: [TF.resized_crop(labels[x][r], i, j, h, w, self.size, self.interpolation)
                             for r in range(len(labels[x]))
                             ]
                         for x in name_list
                         }

        sample = {'image': croped_img,
                  'labels': croped_labels
                  }

        return sample


class HorizontalFlip(object):
    """ HorizontalFlip the given PIL Image
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be

        Returns:
        """
        img, labels = sample['image'], sample['labels']

        img = [TF.hflip(img[r])
               for r in range(len(img))]

        labels = {x: [TF.hflip(labels[r])
                  for r in range(len(labels))
                      ]
                  for x in name_list
                  }

        sample = {'image': img,
                  'labels': labels
                  }

        return sample


class CenterCrop(transforms.CenterCrop):
    """CenterCrop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.CenterCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']

        croped_img = [TF.center_crop(img[r], self.size)
                      for r in range(len(img))]
        croped_labels = {x: [TF.center_crop(labels[r], self.size)
                         for r in range(len(labels))
                             ]
                         for x in name_list
                         }

        sample = {'image': croped_img,
                  'labels': croped_labels
                  }

        return sample


class RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        img_in = []
        for r in range(len(img)):
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img[r].size)
            img_in.append(TF.affine(img[r], *ret, resample=self.resample, fillcolor=self.fillcolor))

        labels = {x: [TF.affine(labels[x][r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                  for r in range(len(labels[x]))]
                  for x in name_list
                  }
        sample = {'image': img,
                  'labels': labels
                  }
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = [TF.to_pil_image(np.uint8(255 * random_noise(np.array(img[r], np.uint8))))
               for r in range(len(img))
               ]
        sample = {'image': img,
                  'labels': labels
                  }

        return sample


class Blurfilter(object):
    # img: PIL image
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = [img[r].filter(ImageFilter.BLUR)
               for r in range(len(img))]
        sample = {'image': img,
                  'labels': labels
                  }

        return sample


