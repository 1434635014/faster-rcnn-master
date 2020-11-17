#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:49:42 2018

@author: jon-liu
"""


import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.engine as KE
import keras.layers as KL

# 切片处理
def batch_slice(inputs, graph_fn, batch_size, names=None):
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
        
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result

# 计算偏移量
def box_refinement_graph(boxes, gt_box):
    boxes = tf.cast(boxes, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    
    heght = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * heght 
    center_x = boxes[:, 1] + 0.5 * width 
    
    gt_h = gt_box[:, 2] - gt_box[:, 0]
    gt_w = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_h 
    gt_center_x = gt_box[:, 1] + 0.5 * gt_w 
    
    dy = (gt_center_y - center_y) / heght
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_h / heght)
    dw = tf.log(gt_w / width)
    deltas = tf.stack([dy, dx, dh, dw], axis=1)
    return deltas
    
# 算两组boxes的iou的
def overlaps_graph(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),[1,1,tf.shape(boxes2)[0]]),[-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    
    intersection = tf.maximum((y2 - y1),0) * tf.maximum((x2 - x1),0)
    union = (b1_y2 - b1_y1) * (b1_x2 - b1_x1) + (b2_y2 - b2_y1) * (b2_x2 - b2_x1) - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps
    
#def trim_zeros_graph(x, name=None):
#    none_zeros = tf.cast(tf.reduce_sum(tf.abs(x), axis=1), tf.bool)
#    result = tf.boolean_mask(x, none_zeros, name=name)
#    return result, none_zeros

# 去掉0元素
def trim_zeros_graph(boxes, name=None):
    none_zero = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes,none_zero, name=name)
    return boxes, none_zero

# proposals：rpn proposals 所提取得分高的结果
# gt_class_ids：真实的分类
# gt_bboxes：真实的边框
# config：配置
def detection_target_graph(proposals, gt_class_ids, gt_bboxes, config):
    # 提取出非0部分
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_bboxes, none_zeros = trim_zeros_graph(gt_bboxes, name="trim_bboxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, none_zeros)
    
    # 算结果框与真实边框的iou
    overlaps = overlaps_graph(proposals, gt_bboxes)
    # 沿着第1维取最大值（从第0维开始）,N*M，取M维
    max_iouArg = tf.reduce_max(overlaps, axis=1)
    # 沿着第0维取最大值，得到下标
    max_iouGT = tf.argmax(overlaps, axis=0) 
    
    # 大于0.5的框（正推荐框）
    positive_mask = (max_iouArg > 0.5)
    # 大于0.5的框的下标
    positive_idxs = tf.where(positive_mask)[:,0]
    # 小于0.5的框的下标（负推荐框）
    negative_idxs = tf.where(max_iouArg < 0.5)[:,0]
    
    # 取多少个推荐框出来，21 * 0.333 = 7
    num_positive = int(config.num_proposals_train *  config.num_proposals_ratio)
    # 取出来的推荐框下标
    positive_idxs = tf.random_shuffle(positive_idxs)[:num_positive]
    # 如果没有推荐的，取的最大值的，至少保证有一个推荐框，并去重得到最终推荐框下标
    positive_idxs = tf.concat([positive_idxs, max_iouGT], axis=0)
    positive_idxs = tf.unique(positive_idxs)[0]
    # 推荐框个数
    num_positive = tf.shape(positive_idxs)[0]
    
    # 取出负的推荐款
    r = 1 / config.num_proposals_ratio
    num_negative = tf.cast(r * tf.cast(num_positive, tf.float32), tf.int32) - num_positive
    negative_idxs = tf.random_shuffle(negative_idxs)[:num_negative]
    
    # 正推荐框
    positive_rois = tf.gather(proposals, positive_idxs)
    # 负推荐框
    negative_rois = tf.gather(proposals, negative_idxs)

    # 正的推荐框iou
    positive_overlap = tf.gather(overlaps, positive_idxs)
    
    # 按一维取正的推荐框iou最大值 的 下标
    gt_assignment = tf.argmax(positive_overlap, axis=1)
    # 按下标取元素，取出与之对应的 真实的boxes
    gt_bboxes = tf.gather(gt_bboxes, gt_assignment)
    # 同上取出分类
    gt_class_ids = tf.gather(gt_class_ids, gt_assignment)
    
    # 计算偏移量（正推荐框 与 真实框）
    deltas = box_refinement_graph(positive_rois, gt_bboxes)
    # 避免数太小
    deltas /= config.RPN_BBOX_STD_DEV
    #deltas = utils.anchor_deltas(positive_rois, gt_bboxes)
    # concat在一起
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    
    # num_proposals_train：训练的个数 21
    N = tf.shape(negative_rois)[0]
    P = config.num_proposals_train - tf.shape(rois)[0]
    
    # 如果不够就padding在一起凑够21个
    rois = tf.pad(rois,[(0,P),(0,0)])
    gt_class_ids = tf.pad(gt_class_ids, [(0, N+P)])
    deltas = tf.pad(deltas,[(0,N+P),(0,0)])
    gt_bboxes = tf.pad(gt_bboxes,[(0,N+P),(0,0)])
    
    return rois, gt_class_ids, deltas, gt_bboxes 
    
    
class DetectionTarget(KE.Layer):
    
    def __init__(self, config, **kwargs):
        super(DetectionTarget, self).__init__(**kwargs)
        self.config = config
        
    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_bboxes = inputs[2]
        
        names = ["rois", "target_class_ids", "target_deltas","target_bbox"]
        outputs = batch_slice([proposals, gt_class_ids, gt_bboxes],
                            lambda x,y,z: detection_target_graph(x, y, z, self.config), self.config.batch_size, names=names)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return [(None, self.config.num_proposals_train, 4),
                (None, 1),
                (None, self.config.num_proposals_train, 4),
                (None, self.config.num_proposals_train, 4)]
                
    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]