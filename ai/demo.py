# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.data import transforms as T

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

logging.disable(logging.CRITICAL)
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def drawn_detections(detections, threshold, cats, im_name, im, K, cfg):
    n_det = len(detections)
    meshes = []
    meshes_text = []

    if n_det > 0:
        for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                detections.pred_bbox3D, detections.pred_center_cam, detections.pred_center_2D, detections.pred_dimensions, 
                detections.pred_pose, detections.scores, detections.pred_classes
            )):

            # skip
            if score < threshold:
                continue
            
            cat = cats[cat_idx]

            bbox3D = center_cam.tolist() + dimensions.tolist()
            meshes_text.append('{} {:.2f}'.format(cat, score))
            color = [c/255.0 for c in util.get_color(idx)]
            box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
            meshes.append(box_mesh)

    print('File: {} with {} detections'.format(im_name, len(meshes)))

    if len(meshes) > 0:
        return vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85, device=cfg.MODEL.DEVICE)
        


def infer_image(im, principal_point, augmentations, model, focal_length, cfg):
    if im is None:
        return False
    
    image_shape = im.shape[:2]  # h, w

    h, w = image_shape
    
    if focal_length == 0:
        focal_length_ndc = 4.0
        focal_length = focal_length_ndc * h / 2

    if len(principal_point) == 0:
        px, py = w/2, h/2
    else:
        px, py = principal_point

    K = np.array([
        [focal_length, 0.0, px], 
        [0.0, focal_length, py], 
        [0.0, 0.0, 1.0]
    ])

    aug_input = T.AugInput(im)
    _ = augmentations(aug_input)
    image = aug_input.image

    batched = [{
        'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(cfg.MODEL.DEVICE), 
        'height': image_shape[0], 'width': image_shape[1], 'K': K
    }]

    detections = model(batched)[0]['instances']

    return detections, K

def do_test(params, cfg, model):

    list_of_ims = ["ai/input.jpg"]

    model.eval()
    
    focal_length = params.focal_length
    principal_point = params.principal_point
    threshold = params.threshold

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    category_path = os.path.join(util.file_parts(params.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    
    for path in list_of_ims:
        im_name = util.file_parts(path)[1]
        im = util.imread(path)

        detections, K = infer_image(im, principal_point, augmentations, model, focal_length, cfg)
        im_drawn_rgb, im_topdown, _ = drawn_detections(detections, threshold, cats, im_name, im, K, cfg)

        if params.display:
            im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
            vis.imshow(im_concat)


def setup(params):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = params.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(params.opts)
    cfg.freeze()
    default_setup(cfg, params)
    return cfg

class Parameters:
    def __init__(self, threshold, focal_length, principal_point, display, model, device):
        self.threshold = threshold
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.display = display

        if model == "DLA":
            self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
            self.opts=['MODEL.WEIGHTS', 'cubercnn://omni3d/cubercnn_DLA34_FPN.pth', 'MODEL.DEVICE', device]
        else:
            self.config_file = "cubercnn://omni3d/cubercnn_Res34_FPN.yaml"
            self.opts=['MODEL.WEIGHTS', 'cubercnn://omni3d/cubercnn_Res34_FPN.pth', 'MODEL.DEVICE', device]

def main(params):
    cfg = setup(params)
    model = build_model(cfg)
    
    DetectionCheckpointer(model).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    with torch.no_grad():
        do_test(params, cfg, model)

if __name__ == "__main__":
    params = Parameters(0.25, 0, [], True, "Res", "cpu")
    main(params)