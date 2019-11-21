# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import PIL
from PIL import ImageDraw

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image, size=None):
        if size is None: size = random.choice(self.min_size)
        if isinstance(image, list) or isinstance(image, tuple):
            return [self.get_size(im, size) for im in image]
        else:
            w, h = image.size
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

    def __call__(self, image, target=None):
        
        if isinstance(image, list) or isinstance(image, tuple):
            sizes = self.get_size(image)
            # TODO: check sizes equal
            image = [F.resize(im, size) for im, size in zip(image, sizes)]
            if target is None:
                return image
            target = target.resize(image[0].size)
        else:
            size = self.get_size(image)
            image = F.resize(image, size)
            if target is None:
                return image
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, list) or isinstance(image, tuple):
                image = [F.hflip(im) for im in image]
            else:
                image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, list) or isinstance(image, tuple):
                image = [F.vflip(im) for im in image]
            else:
                image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class RandomAffineTransform(object):
    def __init__(self, trans_range, scale_range): # TODO: rotate_range
        self.trans_range = trans_range
        self.scale_range = scale_range

    def __call__(self, image, target=None):
        
        t = [(self.trans_range[0] + (self.trans_range[1] - self.trans_range[0]) * random.random()) for i in range(2)]
        s = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * random.random()
        if isinstance(image, list) or isinstance(image, tuple):
            # TODO: check sizes equal
            image = [F.affine(im, 0, (t[0] * im.size[0], t[1] * im.size[1]), s, 0, resample=PIL.Image.BILINEAR) for im in image]
            if target is None:
                return image
            target = target.affine(t, s)
        else:
            # image.save("orig.jpg")
            # print(image.size, t, s)
            image = F.affine(image, 0, (t[0] * image.size[0], t[1] * image.size[1]), s, 0, resample=PIL.Image.BILINEAR)
            if target is None:
                return image
            target = target.affine(t, s)
            # draw = ImageDraw.Draw(image)
            # target = target.convert("xyxy")
            # for box in target.bbox:
            #     draw.rectangle(box.tolist())
            # image.save("out.jpg")
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        if isinstance(image, list) or isinstance(image, tuple):
            image = [self.color_jitter(im) for im in image]
        else:
            image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, list) or isinstance(image, tuple):
            image = [F.to_tensor(im) for im in image]
        else:
            image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if isinstance(image, list) or isinstance(image, tuple):
            images_new = []
            for im in image:
                if self.to_bgr255:
                    im = im[[2, 1, 0]] * 255
                im = F.normalize(im, mean=self.mean, std=self.std)
                images_new.append(im)
            image = images_new
        else:
            if self.to_bgr255:
                image = image[[2, 1, 0]] * 255
            image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
