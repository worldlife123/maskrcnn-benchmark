from .kitti_eval import do_kitti_evaluation


def kitti_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_kitti_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
    )
