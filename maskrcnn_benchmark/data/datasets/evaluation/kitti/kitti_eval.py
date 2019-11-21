import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

import numpy as np
import cv2
from .ddd_utils import ddd2locrot

KITTI_CLASSNAMES =  ['__background__', 'Pedestrian', 'Car', 'Cyclist']

def do_kitti_evaluation(
    dataset,
    predictions,
    output_folder,
    score_threshold=0.05,
    validate=True,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    logger.info("Preparing results for KITTI format")
    # result table "file_name" : result
    kitti_results = {} 
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        file_name = img_info["file_name"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xyxy")
        # print(prediction)
        
        scores = prediction.get_field("scores")
        positive_indices = scores > score_threshold
        scores = scores.tolist()

        boxes = prediction.bbox[positive_indices].tolist()
        labels = prediction.get_field("labels")[positive_indices].tolist()

        # 3D Evaluation
        if prediction.has_field('alphas'):
          centers = prediction.get_field('centers').bbox[:,0:2]
          # centers = prediction.get_field("keypoints").keypoints[:,:,:2]
          centers = centers.view(-1, 2)[positive_indices].tolist()
          depths = prediction.get_field('depths')[positive_indices].tolist()
          dims = prediction.get_field('dims')
          dims = dims.view(-1, 3)[positive_indices].tolist()
          rots = prediction.get_field('rots')
          rots = rots.view(-1, 8)[positive_indices].tolist()
          alphas = prediction.get_field('alphas')[positive_indices].tolist()
        else:
          centers = [[None] * 2] * len(boxes)
          depths = [[None]] * len(boxes)
          dims = [[None] * 3] * len(boxes)
          rots = [[None] * 8] * len(boxes)
          alphas = [None] * len(boxes)

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]  
        
        # locations, rotation_y = ddd2locrot(
        #   center, alpha, dimensions, depth, calibs[0])

        kitti_results[file_name] = []
        
        for k in range(len(boxes)):
            kitti_results[file_name].append({
                'image_id': original_id,
                'calib': img_info['calib'],
                'category_id': mapped_labels[k],
                'bbox': boxes[k],
                'center': centers[k],
                'depth': depths[k],
                "dim": dims[k],
                "rot": rots[k],
                "alpha": alphas[k],
                'score': scores[k],
            })


    logger.info("Evaluating predictions")

    # save results to output folder
    if output_folder:
        # torch.save(results, os.path.join(output_folder, "kitti_results.pth"))
        if not os.path.exists(output_folder): os.mkdir(output_folder)
        for file_name in kitti_results:
            output_file = os.path.join(output_folder, file_name.replace(".png", ".txt"))
            # print(output_file)
            file = open(output_file, 'w')
            save_results(file, kitti_results[file_name])
            file.close()

    # finally, call kitti eval tool (validation only)
    if validate:
      cmd = './tools/kitti/kitti_eval/evaluate_object_3d_offline ' + \
                './datasets/kitti/training/label_2 ' + \
                  output_folder + '/'
      print("Executing: " + cmd)
      os.system(cmd)
    else:
      print("Results saved at " + output_folder)

    return kitti_results

def save_results(f, results):
    for i, det in enumerate(results):
        if det["category_id"] >= len(KITTI_CLASSNAMES): continue # only evaluate Pedestrian, Car, Cyclist 
        class_name = KITTI_CLASSNAMES[det["category_id"]]
        f.write('{} 0.0 0'.format(class_name))
        alpha = det["alpha"] # get_alpha(det["rot"])
        f.write(' {:.2f}'.format(-10 if alpha is None else alpha))
        for num in det["bbox"]:
          f.write(' {:.2f}'.format(num))
        for num in det["dim"]:
          f.write(' {:.2f}'.format(-1 if num is None else num))
        center = [(det["bbox"][0] + det["bbox"][2])/2, (det["bbox"][1] + det["bbox"][3])/2]
        # center = det["center"]
        if det["depth"][0] is None:
          location, rotation_y = [None]*3, None
        else:
          location, rotation_y = ddd2locrot(np.array(center), alpha, np.array(det["dim"]), det["depth"][0], np.array(det["calib"]))  
        for num in location:
          f.write(' {:.2f}'.format(-1000 if num is None else num))
        f.write(' {:.2f}'.format(-10 if rotation_y is None else rotation_y))
        f.write(' {:.2f}'.format(det["score"]))
        if not (i+1) == len(results): f.write('\n')

def get_pred_depth(depth):
    return depth
  
def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  