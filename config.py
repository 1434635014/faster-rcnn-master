#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:32:23 2018

@author: jon-liu
"""

import numpy as np

class Config():
    image_size = [224, 224] # change
    rpn_stride = 16
    featureMap_size = [image_size[0]/rpn_stride, image_size[1]/rpn_stride] # change
    scales = [65, 130, 260] # change
    ratios = [0.5, 1, 2]
    anchor_stride = 1
    train_rois_num = 100
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    num_before_nms = 300
    max_gt_obj = 60  # 原本30
    num_proposals_train = 21
    num_proposals_ratio = 0.333
    batch_size = 10
    img_min_side = image_size[0]
    
    