from detectron2.config import get_cfg
from .predictor import VisualizationDemo
import os
import torch


class PanopticSegmenter():

    def __init__(self, segm_out_type):

        self.segm_out_type = segm_out_type

        self.path_dir = './models/segmentation/detectron2'
        config_dir = "configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
        self.config_file = os.path.join(self.path_dir, config_dir)
        weight_file = 'model_final_c10459_panoptic.pkl'
        self.opts = ['MODEL.WEIGHTS', os.path.join(self.path_dir, weight_file)]
        cfg = self.setup_cfg()

        self.demo = VisualizationDemo(cfg)

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(self.opts)

        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.4
        cfg.freeze()
        return cfg

    def segment_images(self, images):
        """
        :param images: tensor image
        :return:
        """

        all_mask, all_mask_sep = self.demo.run_on_image(images)

        return torch.from_numpy(all_mask).permute(2, 0, 1).float().cuda().unsqueeze(0),\
               torch.from_numpy(all_mask_sep).permute(2, 0, 1).float().cuda().unsqueeze(0)

