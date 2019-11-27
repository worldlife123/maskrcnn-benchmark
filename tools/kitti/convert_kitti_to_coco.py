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
import transformations
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

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    # each row of the velodyne data is forward, left, up, reflectance
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    # remove all behind image plane (approximation)
    points = points[points[:, 0] >= 0, :]
    return points

def transform_points_to_camera(points, calibs):
    # project the points to the camera
    velo2cam = np.vstack((calibs['Tr_velo_to_cam'], np.array([0, 0, 0, 1.0])))
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = calibs['R0_rect']
    velo2cam = np.dot(R_cam2rect, velo2cam)
    velo_pts_cam = np.dot(velo2cam, points.T).T

    return velo_pts_cam

def transform_points_to_image(points, calibs, im_shape, cam_id=2, vel_depth=False):
    # project the points to the camera
    velo2cam = np.vstack((calibs['Tr_velo_to_cam'], np.array([0, 0, 0, 1.0])))
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = calibs['R0_rect']
    P_rect = calibs['P'+str(cam_id)] # TODO: different camera
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    velo_pts_im = np.dot(P_velo2im, points.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = points[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[0]) & (velo_pts_im[:, 1] < im_shape[1])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    # depth = np.zeros((im_shape[:2]))
    # depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    # def _sub2ind(matrixSize, rowSub, colSub):
    #   """Convert row, col matrix subscripts to linear indices
    #   """
    #   m, n = matrixSize
    #   return rowSub * (n-1) + colSub - 1
    # inds = _sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    # dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    # for dd in dupe_inds:
    #     pts = np.where(inds == dd)[0]
    #     x_loc = int(velo_pts_im[pts[0], 0])
    #     y_loc = int(velo_pts_im[pts[0], 1])
    #     depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    # depth[depth < 0] = 0

    return velo_pts_im

def filter_points_inside_box(points, dim, loc, ry):
  box_T = transformations.translation_matrix(loc)
  box_R = transformations.rotation_matrix(ry, [0,1,0])
  box_S = np.diag([dim[2], dim[0], dim[1], 1.0]) # transformations.scale_matrix(dim)
  inv_mat = np.linalg.inv(np.dot(box_T, np.dot(box_R, box_S))) # post multiply
  points_box_trans = np.dot(inv_mat, points.T).T
  inside_idxs = (points_box_trans[:, 0] >= -0.5) & (points_box_trans[:, 0] <= 0.5) & \
    (points_box_trans[:, 1] >= -0.5) & (points_box_trans[:, 1] <= 0.5) & \
    (points_box_trans[:, 2] >= -0.5) & (points_box_trans[:, 2] <= 0.5)
  return points_box_trans[inside_idxs], inside_idxs


def get_bbox_from_image_points(points):
  return [float(np.min(points[:,0])), float(np.min(points[:,1])), float(np.max(points[:,0])), float(np.max(points[:,1]))]

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
  velodyne_dir = DATA_PATH + 'training/velodyne/'
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
      # read png header
      img_file = open(os.path.join(image_dir + '{}.png'.format(line)), 'rb')
      img_header = img_file.read(25)
      w, h = struct.unpack('>LL', img_header[16:24])
      width = int(w)
      height = int(h)
      # print(width, height)
      img_file.close()
      baseline = ((calib[0,3] - calibs["P3"][0,3]) / calib[0,0]).item()
      image_info = {'file_name': '{}.png'.format(line),
                    'right_file_name': '../image_3/{}.png'.format(line),
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
                        "baseline": baseline
                      },
                    },
                    'calib': calib.tolist()}
      ret['images'].append(image_info)

      # process velodyne
      points = load_velodyne_points(os.path.join(velodyne_dir + '{}.bin'.format(line)))
      # points_image = transform_points_to_image(points, calibs, (width, height))

      if split == 'test':
        continue
      ann_path = ann_dir + '{}.txt'.format(line)
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      anns = open(ann_path, 'r')
      
      if DEBUG:
        image = cv2.imread(
          image_dir + image_info['file_name'])
        image_right = cv2.imread(
          image_dir + image_info['right_file_name'])

      print("Processing %s" % ann_path)

      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        if tmp[0] == 'DontCare': continue # filter out (TODO:)
        truncated = float(tmp[1])
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])

        location_center = np.array(location)
        location_center[1] -= dim[0]/2

        coco_bbox = _bbox_to_coco_bbox(bbox)

        box_3d = compute_box_3d(dim, location, rotation_y)
        box_2d = project_to_image(box_3d, calib)
        box_2d_right = project_to_image(box_3d, calibs["P3"])
        bbox_from_3d = get_bbox_from_image_points(box_2d)
        bbox_right_from_3d = get_bbox_from_image_points(box_2d_right)
        disp_from_3d = (bbox_from_3d[2] + bbox_from_3d[0] - bbox_right_from_3d[2] - bbox_right_from_3d[0]) / 2
        depth_from_3d = baseline * calib[0,0] / disp_from_3d
        

        # adjust box and disparity with velodyne points
        points_cam = transform_points_to_camera(points, calibs)
        points_inside_box, pidxs = filter_points_inside_box(points_cam, dim, location_center, rotation_y)
        points_img = transform_points_to_image(points[pidxs], calibs, (width, height), cam_id=2)
        points_img_right = transform_points_to_image(points[pidxs], calibs, (width, height), cam_id=3)
        # print(points[pidxs].shape)
        # print(len(points_img), len(points_img_right))
        if len(points_img) == 0 or len(points_img_right) == 0: 
          print("WARNING! Cannot find points inside box")
          # continue
          # use box location and dimension to estimate depth
          depth_adjust = location[2] - (np.sin(rotation_y)*dim[2]+np.cos(rotation_y)*dim[1])/2
          # coco_bbox_adjust = None # coco_bbox
          # coco_bbox_adjust_right = None # coco_bbox
        else:

          sorted_depth = np.sort(points_img[:,2])
          depth_adjust = sorted_depth[len(sorted_depth)//10]
          # print(depth_adjust, depth_from_3d, location[2])

        # bbox_adjust = get_bbox_from_image_points(points_img)
        bbox_adjust = get_bbox_from_image_points(box_2d)
        coco_bbox_adjust = _bbox_to_coco_bbox(bbox_adjust)

        # bbox_adjust_right = get_bbox_from_image_points(points_img_right)
        bbox_adjust_right = get_bbox_from_image_points(box_2d_right)
        coco_bbox_adjust_right = _bbox_to_coco_bbox(bbox_adjust_right)
        # print(bbox, bbox_adjust, bbox_adjust_right)
        
        # print(bbox, project_to_image(np.array([location]), calib))
        # if depth_adjust<0: print(depth_adjust)
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
               'rotation_y': rotation_y,
               'bbox_adjust': None if coco_bbox_adjust is None else coco_bbox_adjust,
               'bbox_adjust_right': None if coco_bbox_adjust_right is None else coco_bbox_adjust_right,
               'depth_adjust': None if depth_adjust is None else depth_adjust,
               'depth_from_3d': depth_from_3d,
               }
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d)
          image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0))
          image = cv2.rectangle(image, (int(bbox_adjust[0]), int(bbox_adjust[1])), (int(bbox_adjust[2]), int(bbox_adjust[3])), color=(0,255,0))
          # image = cv2.rectangle(image, (int(bbox_adjust_right[0]), int(bbox_adjust_right[1])), (int(bbox_adjust_right[2]), int(bbox_adjust_right[3])), color=(0,0,255))
          # for pt in transform_points_to_image(points, calibs, (width, height), cam_id=2):
          #   image = cv2.circle(image, (int(pt[0]), int(pt[1])), radius=1, color=(0,0,255))
          for pt in points_img:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), radius=2, color=(0,255,0))
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          box_2d = project_to_image(box_3d, calibs["P3"])
          # print('box_2d', box_2d)
          # disp =  baseline * calib[0,0].item() / depth_adjust
          # bbox_right = [bbox[0]-disp, bbox[1], bbox[2]-disp, bbox[3]]
          image_right = draw_box_3d(image_right, box_2d)
          image_right = cv2.rectangle(image_right, (int(bbox_adjust_right[0]), int(bbox_adjust_right[1])), (int(bbox_adjust_right[2]), int(bbox_adjust_right[3])), color=(255,0,0))
          
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                            dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2
          print('pt_3d', pt_3d)
          print('location', location)
      if DEBUG:
        cv2.imshow('image', image)
        cv2.imshow('image_right', image_right)
        cv2.waitKey()
      # break


    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
    json.dump(ret, open(out_path, 'w'))
    out_path_adjust = '{}/annotations/kitti_{}_{}_box_from_3d.json'.format(DATA_PATH, SPLIT, split)
    for anno in ret['annotations']:
      anno["bbox"] = anno.pop("bbox_adjust")
      anno["bbox_right"] = anno.pop("bbox_adjust_right")
    json.dump(ret, open(out_path_adjust, 'w'))
  
