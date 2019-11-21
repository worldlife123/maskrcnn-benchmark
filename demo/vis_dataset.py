
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODatasetDemo

import numpy as np

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Do not show visualization",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODatasetDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        test_only=args.test_only,
    )
    # coco_demo.next_data(440)
    
    while True:
        start_time = time.time()
        try:
            composite, composite_gt = coco_demo.next_data()
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            if not args.test_only:
                cv2.imshow("COCO detections", composite)
                cv2.imshow("Ground Truth", composite_gt)
                key = cv2.waitKey(-1) 
                # print(key)
                if key == 27:
                    break  # esc to quit
                elif key == ord('s'):
                    print("Saving..")
                    cv2.imwrite("tmp.jpg", composite)
                    cv2.imwrite("tmp_gt.jpg", composite_gt)
        except StopIteration:
            break

    # print error stats
    error_stats = coco_demo.error_stats
    abs_errs = np.array([stat["abs_err"] for stat in error_stats])
    rel_errs = np.array([stat["rel_err"] for stat in error_stats])
    gts = np.array([stat["gt"] for stat in error_stats])
    gt_close_idx = gts <= 20
    gt_medium_idx = (gts > 20) & (gts <= 50)
    gt_far_idx = gts > 50
    print("Close (depth<20m) Absolute Error: %s" % (np.average(abs_errs[gt_close_idx])))
    print("Close (depth<20m) Relative Error: %s" % (np.average(rel_errs[gt_close_idx])))
    print("Medium (depth<50m) Absolute Error: %s" % (np.average(abs_errs[gt_medium_idx])))
    print("Medium (depth<50m) Relative Error: %s" % (np.average(rel_errs[gt_medium_idx])))
    print("Far Absolute Error: %s" % (np.average(abs_errs[gt_far_idx])))
    print("Far Relative Error: %s" % (np.average(rel_errs[gt_far_idx])))
    print("Total Absolute Error: %s" % (np.average(abs_errs)))
    print("Total Relative Error: %s" % (np.average(rel_errs)))
    


    cv2.destroyAllWindows()

    # print error information


if __name__ == "__main__":
    main()
