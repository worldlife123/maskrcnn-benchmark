from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .coco import coco_evaluation_with_depth
from .voc import voc_evaluation
from .kitti import kitti_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.DukeMTMCDataset) or isinstance(dataset, datasets.CityScapesWDDataset) or isinstance(dataset, datasets.CityScapesWHDataset)  or isinstance(dataset, datasets.CityScapesLRDataset):
        return coco_evaluation_with_depth(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.KITTI3DDataset):
        return kitti_evaluation(**args)
        # return coco_evaluation_with_depth(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
