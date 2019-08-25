from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
DATA_PATH = '../../datasets/kitti/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os, struct
SPLITS = ['3dop', 'subcnn'] 
# import _init_paths
from ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_calib_all(calib_path):
  f = open(calib_path, 'r')
  calibs = {}
  for i, line in enumerate(f):
    # if i == 2:
    segs = line[:-1].split(' ')
    calib = np.array(segs[1:], dtype=np.float32)
    calib = calib.reshape(3, -1)
    calibs[segs[0][:-1]] = calib
  return calibs

def read_clib(calib_path):
  f = open(calib_path, 'r')
  for i, line in enumerate(f):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384 # 375
W = 1248 # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]], 
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
  image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLIT)
  ann_dir = DATA_PATH + 'training/label_2/'
  calib_dir = DATA_PATH + '{}/calib/'
  image_dir = DATA_PATH + 'training/image_2/'
  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                'test': 'testing'}

  for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    image_set = open(image_set_path + '{}.txt'.format(split), 'r')
    image_to_id = {}
    for line in image_set:
      if line[-1] == '\n':
        line = line[:-1]
      image_id = int(line)
      calib_path = calib_dir.format(calib_type[split]) + '{}.txt'.format(line)
      # calib = read_clib(calib_path)
      calibs = read_calib_all(calib_path)
      calib = calibs["P2"]
      # image = cv2.imread(os.path.join(image_dir + '{}.png'.format(line))) # TODO: time-consuming
      img_file = open(os.path.join(image_dir + '{}.png'.format(line)), 'rb')
      img_header = img_file.read(25)
      # read png header
      w, h = struct.unpack('>LL', img_header[16:24])
      width = int(w)
      height = int(h)
      print(width, height)
      img_file.close()
      image_info = {'file_name': '{}.png'.format(line),
                    'id': int(image_id),
                    "width": width, # image.shape[1]
                    "height": height, # image.shape[0]
                    "camera_params": {
                      "intrinsic":{
                        "fx":calib[0,0].item(),
                        "fy":calib[1,1].item(),
                        "cx":calib[0,2].item(),
                        "cy":calib[1,2].item(),
                      },
                      "extrinsic":{
                        "baseline": ((calib[0,3] - calibs["P3"][0,3]) / calib[0,0]).item()
                      },
                    },
                    'calib': calib.tolist()}
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      anns = open(ann_path, 'r')
      
      if DEBUG:
        image = cv2.imread(
          image_dir + image_info['file_name'])

      print("Processing %s" % ann_path)

      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        if tmp[0] == 'DontCare': continue # filter out (TODO:)
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])

        location_center = np.array(location)
        location_center[1] -= dim[0]/2

        coco_bbox = _bbox_to_coco_bbox(bbox)
        # print(bbox, project_to_image(np.array([location]), calib))
        ann = {'image_id': image_id,
               'id': int(len(ret['annotations']) + 1),
               'category_id': cat_id,
               "iscrowd": 0,
               'dim': dim,
               'bbox': coco_bbox,
               'area': coco_bbox[2]*coco_bbox[3],
               'box_center': project_to_image(np.array([location_center]), calib)[0].tolist(),
               'depth': location[2],
               'alpha': alpha,
               'truncated': truncated,
               'occluded': occluded,
               'location': location,
               'rotation_y': rotation_y}
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0))
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2
          print('pt_3d', pt_3d)
          print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.waitKey()


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'))
  
