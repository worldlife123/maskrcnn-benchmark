# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
import torch

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if isinstance(transposed_batch[0][0], dict):
            images = {}
            for image_dict in transposed_batch[0]:
                for k,v in image_dict.items():
                    if not images.get(k): images[k] = []
                    images[k].append(v)
            if images.get("img_info"):
                img_info = images.pop("img_info")
            else:
                img_info = None
            images = {k: to_image_list(v, self.size_divisible, img_info) for k,v in images.items()}
        else:
            images = to_image_list(transposed_batch[0], self.size_divisible)
        if isinstance(transposed_batch[1][0], dict):
            targets = {}
            for target_dict in transposed_batch[1]:
                for k,v in target_dict.items():
                    if not targets.get(k): targets[k] = []
                    targets[k].append(v)
            targets = {k: v for k,v in targets.items()}
        else:
            targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

class BatchCollator2Input(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[2]
        images2 = to_image_list(transposed_batch[1], self.size_divisible)
        targets2 = transposed_batch[3]
        img_ids = transposed_batch[4]
        return images, images2, targets, targets2, img_ids

class BatchCollatorSelfAdapt(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        transposed_batch = [(to_image_list(b, self.size_divisible) if isinstance(b[0], torch.Tensor) else b) for b in transposed_batch ]
        # images = to_image_list(transposed_batch[0], self.size_divisible)
        # targets = transposed_batch[1]
        # img_ids = transposed_batch[2]
        # return images, targets, img_ids
        return tuple(transposed_batch)

class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

