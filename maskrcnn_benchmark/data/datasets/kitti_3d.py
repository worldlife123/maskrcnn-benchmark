# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints, BoxCenterKeypoints
from maskrcnn_benchmark.structures.point_3d import PointDepth

import math

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


class KITTI3DDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, is_train, 
        class_filter_list=[], 
        remove_truncated=False, 
        transforms=None, 
        depth_key="depth", 
        input_depth_mode="depth",
        output_depth_mode="depth", 
        depth_range=(0,0)
    ):
        super(KITTI3DDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        self.is_train = is_train

        self.class_filter_list = [cat['id'] for cat in self.coco.cats.values() if cat['name'] in class_filter_list]

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        print(self.categories)

        catIds = self.class_filter_list if len(class_filter_list)>0 else self.coco.getCatIds()
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(catIds)
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.remove_images_without_annotations = is_train
        if self.remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno) and len([obj for obj in anno if obj["category_id"] in self.class_filter_list])>0:
                    ids.append(img_id)
            self.ids = ids

        self.remove_truncated = remove_truncated

        self.depth_key = depth_key
        self.input_depth_mode = input_depth_mode
        self.output_depth_mode = output_depth_mode
        self.depth_range = depth_range

        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(KITTI3DDataset, self).__getitem__(idx)

        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.loadImgs(img_id)[0]

        # filter crowd annotations
        # TODO might be better to add an extra field
        if self.remove_truncated:
            anno = [obj for obj in anno if obj["truncated"] == 0]

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

        if anno and self.depth_key in anno[0]:
            depth = [obj[self.depth_key] for obj in anno]
            # depth = torch.tensor(depth)
            depth = PointDepth(depth, img.size, 
                focal_length=img_info["camera_params"]["intrinsic"]["fx"], 
                baseline=img_info["camera_params"]["extrinsic"]["baseline"], 
                min_value=self.depth_range[0],
                max_value=self.depth_range[1],
                mode=self.input_depth_mode)
            depth = depth.convert(self.output_depth_mode)
            target.add_field("depths", depth)

        if anno and "dim" in anno[0]:
            dim = [obj["dim"] for obj in anno]
            dim = torch.tensor(dim)
            target.add_field("dims", dim)

        if anno and "box_center" in anno[0]:
            center = [[obj["box_center"]]+[obj["box_center"]] for obj in anno]
            center_box = torch.tensor(center).reshape(-1, 4)
            # add center as a boxlist so that it can be resized
            target.add_field("centers", BoxList(center_box, img.size, mode="xyxy"))
            # add as keypoints
            keypoints = [obj["box_center"]+[1.] for obj in anno]
            keypoints = BoxCenterKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)


        if anno and "location" in anno[0]:
            location = [obj["location"] for obj in anno]
            location = torch.tensor(location)
            target.add_field("locations", location)

        if anno and "rotation_y" in anno[0]:
            ry = [obj["rotation_y"] for obj in anno]
            ry = torch.tensor(ry)
            target.add_field("rys", ry)

        if anno and "alpha" in anno[0]:
            alphas = [obj["alpha"] for obj in anno]
            # convert alpha to binary-scale orientation (from CenterNet)
            rotbin = torch.zeros(len(alphas), 2, dtype=torch.long)
            rotres = torch.zeros(len(alphas), 2)
            for k,alpha in enumerate(alphas):
                if alpha < math.pi / 6. or alpha > 5 * math.pi / 6.:
                    rotbin[k,0] = 1
                    rotres[k,0] = alpha - (-0.5 * math.pi)    
                if alpha > -math.pi / 6. or alpha < -5 * math.pi / 6.:
                    rotbin[k,1] = 1
                    rotres[k,1] = alpha - (0.5 * math.pi)
            # ori = torch.tensor(alpha)
            alphas = torch.tensor(alphas)
            target.add_field("alphas", alphas)
            target.add_field("rotbins", rotbin)
            target.add_field("rotregs", rotres)
            

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
