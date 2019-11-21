import logging
import tempfile
import os, math
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.point_3d import PointDepth

def _height_to_depth(prediction, img_info):
    boxes = prediction.convert("xywh").bbox
    labels = prediction.get_field("labels")
    heights = prediction.get_field('depths')
    new_depths = heights / (boxes[:, 3]) * img_info["camera_params"]["intrinsic"]["fy"]
    new_depths = PointDepth(new_depths, (img_info["width"], img_info["height"]), 
                focal_length=img_info["camera_params"]["intrinsic"]["fx"], 
                baseline=img_info["camera_params"]["extrinsic"]["baseline"], 
                mode="depth")
    prediction.add_field("depths", new_depths)
    return prediction


def depth_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    score_threshold=0.05,
    bbox_iou_threshold=0.5,
    height_to_depth=False,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    logger.info("Preparing results for Depth Evaluation")
    # result table "file_name" : result
    depth_results = {} 
    gt_box_num = 0
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        file_name = img_info["file_name"]

        # ground truth
        # img, gt, idx = dataset[original_id] # TODO: load gt only
        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if hasattr(dataset, 'remove_truncated') and dataset.remove_truncated:
            anno = [obj for obj in anno if obj["truncated"] == 0]

        if hasattr(dataset, 'class_filter_list') and len(dataset.class_filter_list) > 0:
            anno = [obj for obj in anno if obj["category_id"] in dataset.class_filter_list]

        depth_key = dataset.depth_key if hasattr(dataset, 'depth_key') else "depth"
        input_depth_mode = dataset.input_depth_mode if hasattr(dataset, 'input_depth_mode') else depth_key
        output_depth_mode = dataset.output_depth_mode if hasattr(dataset, 'output_depth_mode') else "depth"
        min_value = dataset.min_value if hasattr(dataset, 'min_value') else 0.1
        max_value = dataset.max_value if hasattr(dataset, 'max_value') else 100

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (image_width, image_height), mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [dataset.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if height_to_depth:
            height = [obj["height_rw"] for obj in anno]
            height = torch.tensor(height)
            target.add_field("depths", height)
            target = _height_to_depth(target, img_info)
        elif anno and depth_key in anno[0]:
            depth = [obj[depth_key] for obj in anno]
            # depth = torch.tensor(depth)
            depth = PointDepth(depth, (image_width, image_height), 
                focal_length=img_info["camera_params"]["intrinsic"]["fx"], 
                baseline=img_info["camera_params"]["extrinsic"]["baseline"], 
                min_value=min_value,
                max_value=max_value,
                mode=input_depth_mode)
            target.add_field("depths", depth)

        gt = target.resize((image_width, image_height))

        gt_boxes = gt.bbox.tolist()
        if len(gt_boxes)==0: continue
        gt_box_num += len(gt_boxes)
        gt_labels = gt.get_field("labels").tolist()
        gt_depths = gt.get_field('depths').convert("depth").depths.tolist()
        # print(gt_depths)
        gt_mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in gt_labels]

        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xyxy")
        # print(prediction)

        scores = prediction.get_field("scores")
        positive_indices = scores > score_threshold
        scores = scores.tolist()

        boxes = prediction.bbox[positive_indices].tolist()
        if len(boxes) == 0: continue
        labels = prediction.get_field("labels")[positive_indices].tolist()

        if height_to_depth:
            prediction = _height_to_depth(prediction, img_info)
        depths = prediction.get_field('depths')[positive_indices] # .convert("depth").depths
        if isinstance(depths, PointDepth):
            depths = depths#.convert(output_depth_mode)
        else:
            depths = PointDepth(depths, (image_width, image_height), 
                focal_length=img_info["camera_params"]["intrinsic"]["fx"], 
                baseline=img_info["camera_params"]["extrinsic"]["baseline"], 
                min_value=min_value,
                max_value=max_value,
                mode="depth")
        depths = depths.convert("depth")
        depths = depths.depths.tolist()
        # print(depths, gt_depths)

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]  

        # find corresponding box
        overlaps = boxlist_iou(prediction[positive_indices], gt)
        gt_overlaps = torch.zeros(len(gt_boxes))
        dt_matches = [-1] * len(boxes)
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            if gt_ovr < bbox_iou_threshold: continue
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            dt_matches[box_ind] = gt_ind
            # record the iou coverage of this gt box
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr            
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        
        # locations, rotation_y = ddd2locrot(
        #   center, alpha, dimensions, depth, calibs[0])

        depth_results[file_name] = []
        # gt[file_name] = {}
        
        for k in range(len(boxes)):
            depth_results[file_name].append({
                'image_id': original_id,
                # 'calib': img_info['calib'],
                'category_id': mapped_labels[k],
                'bbox': boxes[k],
                'depth': depths[k][0],
                'gt_category_id': gt_mapped_labels[dt_matches[k]] if dt_matches[k]>=0 else None,
                'gt_bbox': gt_boxes[dt_matches[k]] if dt_matches[k]>=0 else None,
                'gt_depth': gt_depths[dt_matches[k]]  if dt_matches[k]>=0 else None,
                'score': scores[k],
            })
        
        # for k in range(len(gt_boxes)):
        #     gt[file_name].append({
        #         'image_id': original_id,
        #         'calib': img_info['calib'],
        #         'category_id': gt_mapped_labels[k],
        #         'bbox': gt_boxes[k],
        #         'depth': gt_depths[k],
        #     })

    logger.info("Evaluating predictions")
    logger.info("Ground Truth boxes %d" % gt_box_num)
    results = evaluate_results(depth_results)
    import json
    logger.info(json.dumps(results, sort_keys=True, indent=4))

    return results

    # return depth_results


def evaluate_results(
        results, 
        alp_thresholds=[0.5, 1.0, 2.0, 5.0],
        rlp_thresholds=[0.05, 0.1, 0.2, 0.25],
    ):

    eval_results = {
        "Total": 0,
        "Valid": 0,
        "ALE": {},
        "RLE": {},
        "RMSE": {},
        "ALP": {},
        "RLP": {},
    }

    depth_errors = []
    depth_errors_per_class = {}
    depth_rel_errors = []
    depth_rel_errors_per_class = {}
    for file_name, predictions in results.items():
        for prediction in predictions:
            if prediction['gt_depth'] is None or prediction['category_id'] != prediction['gt_category_id']:
                depth_errors.append(-1)
                depth_rel_errors.append(-1)
                if not depth_errors_per_class.get(prediction['category_id']): depth_errors_per_class[prediction['category_id']] = []
                depth_errors_per_class[prediction['category_id']].append(-1)
                if not depth_rel_errors_per_class.get(prediction['category_id']): depth_rel_errors_per_class[prediction['category_id']] = []
                depth_rel_errors_per_class[prediction['category_id']].append(-1)
            else:
                # print(prediction['depth'],prediction['gt_depth'])
                err = abs(prediction['depth']-prediction['gt_depth'])
                rel_err = abs(prediction['depth']-prediction['gt_depth'])/prediction['gt_depth']
                depth_errors.append(err)
                depth_rel_errors.append(rel_err)
                if not depth_errors_per_class.get(prediction['category_id']): depth_errors_per_class[prediction['category_id']] = []
                depth_errors_per_class[prediction['category_id']].append(err)
                if not depth_rel_errors_per_class.get(prediction['category_id']): depth_rel_errors_per_class[prediction['category_id']] = []
                depth_rel_errors_per_class[prediction['category_id']].append(rel_err)

    # ALE, RLE, RMSE
    def _valid(errs):
        valid_error_num = 0
        for err in errs:
            if err>0: 
                valid_error_num += 1
        return valid_error_num

    def _le(errs):
        valid_error_num = 0
        total_error = 0
        for err in errs:
            if err>0: 
                valid_error_num += 1
                total_error += err
        return total_error / valid_error_num if valid_error_num > 0 else 0

    def _mse(errs):
        valid_error_num = 0
        total_error = 0
        for err in errs:
            if err>0: 
                valid_error_num += 1
                total_error += (err*err)
        return math.sqrt(total_error / valid_error_num) if valid_error_num > 0 else 0

    eval_results["ALE"]["Valid"] = _le(depth_errors)
    eval_results["RLE"]["Valid"] = _le(depth_rel_errors)
    eval_results["RMSE"]["Valid"] = _mse(depth_errors)
    # Per Class
    total_error_per_class = {}
    total_rel_error_per_class = {}
    total_mse_per_class = {}
    for cls_id, errs in depth_errors_per_class.items():
        total_error_per_class[cls_id] = _le(errs)
    eval_results["ALE"]["PerClass"] = total_error_per_class
    for cls_id, errs in depth_rel_errors_per_class.items():
        total_rel_error_per_class[cls_id] = _le(errs)
    eval_results["RLE"]["PerClass"] = total_rel_error_per_class
    for cls_id, errs in depth_errors_per_class.items():
        total_mse_per_class[cls_id] = _mse(errs)
    eval_results["RMSE"]["PerClass"] = total_mse_per_class


    # ALP, RLP
    def _lp(errs, thresholds):
        under_threshold_nums = [0] * len(thresholds)
        valid_num = 0
        total_num = 0
        ret = {}
        for err in errs:
            if err>0:
                for i, thr in enumerate(thresholds):
                    if err < thr:
                        under_threshold_nums[i] += 1
                valid_num += 1
            total_num += 1
        # print("total_num: %d" % total_num)
        # print("valid_num: %d" % valid_num)
        ret["Total"] = {thr: (pr+0.0)/total_num for thr, pr in zip(thresholds, under_threshold_nums)}
        ret["Valid"] = {thr: (pr+0.0)/valid_num for thr, pr in zip(thresholds, under_threshold_nums)}
        return ret

    eval_results["ALP"] = _lp(depth_errors, alp_thresholds)
    eval_results["RLP"] = _lp(depth_rel_errors, rlp_thresholds)

    # Per Class
    total_error_per_class = {}
    total_rel_error_per_class = {}
    for cls_id, errs in depth_errors_per_class.items():
        total_error_per_class[cls_id] = _lp(errs, alp_thresholds)
    eval_results["ALP"]["PerClass"] = total_error_per_class
    for cls_id, errs in depth_rel_errors_per_class.items():
        total_rel_error_per_class[cls_id] = _lp(errs, rlp_thresholds)
    eval_results["RLP"]["PerClass"] = total_rel_error_per_class

    eval_results["Total"] = len(depth_errors)
    eval_results["Valid"] = _valid(depth_errors)

    # print(eval_results)
    # import json
    # print(json.dumps(eval_results, sort_keys=True, indent=4))

    return eval_results  