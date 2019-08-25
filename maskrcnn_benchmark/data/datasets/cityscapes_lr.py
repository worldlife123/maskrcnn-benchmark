# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

import os
from PIL import Image

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class CityScapesLRDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(CityScapesLRDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        self.remove_images_without_annotations = remove_images_without_annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        # img, anno = super(CityScapesLRDataset, self).__getitem__(idx)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        right_path = coco.loadImgs(img_id)[0]['right_file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        right_img = Image.open(os.path.join(self.root, right_path)).convert('RGB')

        if self.transform is not None:
            # img = self.transform(img)
            # right_img = self.transform(right_img)
            (img, right_img) = self.transform((img, right_img))

        if self.target_transform is not None:
            anno = self.target_transform(anno)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        boxes_right = [obj["bbox_right"] for obj in anno]
        boxes_right = torch.as_tensor(boxes_right).reshape(-1, 4)  # guard against no boxes
        right_target = BoxList(boxes_right, right_img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        right_target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        # if anno and "height_rw" in anno[0]:
        #     depth = [obj["height_rw"] for obj in anno]
        if anno and "disp_unity" in anno[0]:
            depth = [obj["disp_unity"] for obj in anno]
            depth = torch.tensor(depth)
            target.add_field("depths", depth)
            right_target.add_field("depths", depth)

        target = target.clip_to_image(remove_empty=True)
        right_target = right_target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
            right_img, right_target = self._transforms(right_img, right_target)

        # samples = {
        #     "images_left" : img, 
        #     "images_right" : right_img, 
        #     "targets_left" : target, 
        #     "targets_right" :right_target, 
        #     "idx" : idx
        # }

        # tmp fix to enable evaluation
        if self.remove_images_without_annotations:
            return img, right_img, target, right_target, idx
        else:
            return img, target, idx
        # return samples

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
