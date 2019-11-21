# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.structures.point_3d import PointDepth

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


class CityScapesWDDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, is_train, 
        transforms=None, 
        depth_key="depth", 
        output_depth_mode="disp_base_unity", 
        depth_range=(0,0), 
        class_filter_list=[]
    ):
        '''
        output_depth_mode: this mode should be forced "disp_base_unity" when image size augmentation is applied to the dataset
        '''
        super(CityScapesWDDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        self.class_filter_list = [cat['id'] for cat in self.coco.cats.values() if cat['name'] in class_filter_list]

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        catIds = self.class_filter_list if len(class_filter_list)>0 else self.coco.getCatIds()
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(catIds)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # filter images without detection annotations
        self.is_train = is_train
        if is_train:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    if len(self.class_filter_list) == 0 or len([obj for obj in anno if obj["category_id"] in self.class_filter_list])>0:
                        ids.append(img_id)
            self.ids = ids

        self._transforms = transforms
        self.depth_key = depth_key
        self.output_depth_mode = output_depth_mode
        self.depth_range = depth_range

    def __getitem__(self, idx):
        # img, anno = super(CityScapesWDDataset, self).__getitem__(idx)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        # right_path = img_info['right_file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # right_img = Image.open(os.path.join(self.root, right_path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            # right_img = self.transform(right_img)

        if self.target_transform is not None:
            anno = self.target_transform(anno)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        if len(self.class_filter_list) > 0:
            anno = [obj for obj in anno if obj["category_id"] in self.class_filter_list]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

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
        if anno and self.depth_key in anno[0]:
            depth = [obj[self.depth_key] for obj in anno]
            # depth = torch.tensor(depth)
            depth = PointDepth(depth, img.size, 
                focal_length=img_info["camera_params"]["intrinsic"]["fx"], 
                baseline=img_info["camera_params"]["extrinsic"]["baseline"], 
                min_value=self.depth_range[0],
                max_value=self.depth_range[1],
                mode=self.depth_key)
            # print(depth.depths)
            depth = depth.convert(self.output_depth_mode)
            # print(depth.depths)
            target.add_field("depths", depth)

        target = target.clip_to_image(remove_empty=True)

        # target.add_field("right_image", right_img)

        # print(target.get_field("depths").depths)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # print(target.get_field("depths").depths)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
