#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:37:14 2018

@author: jon-liu
"""


import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.engine as KE
import keras.layers as KL


# 将anchors根据计算出的修正量逆运算出实际计算的anchors
# anchors：锚框
# deltas：网络计算出的修正量
def anchor_refinement(boxes, deltas):
    boxes = tf.cast(boxes, tf.float32)
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + h / 2
    center_x = boxes[:, 1] + w / 2

    center_y += deltas[:, 0] * h
    center_x += deltas[:, 1] * w
    h *= tf.exp(deltas[:, 2])
    w *= tf.exp(deltas[:, 3])
    
    y1 = center_y - h / 2
    x1 = center_x - w / 2
    y2 = center_y + h / 2
    x2 = center_x + w / 2
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    return boxes
    
# 限定坐标的边界，不超过边界
def boxes_clip(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    cliped = tf.concat([y1, x1, y2, x2], axis=1)
    cliped.set_shape((cliped.shape[0], 4))
    return cliped

# 将输入的inputs 按提供的 graph_fn方法进行切片
# 因为数据是batch_size个的数组，所以这个函数是进行batch_size个数组的操作
def batch_slice(inputs, graph_fn, batch_size):
    if not isinstance(inputs, list):
        inputs = [inputs]
    output = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (list, tuple)):
            output_slice = [output_slice]
        output.append(output_slice)
    # 打包输出出去
    output = list(zip(*output))
    result = [tf.stack(o, axis=0) for o in output]
    if len(result)==1:
        result = result[0]
    return result
    
# proposal_count：保留的框个数
# nms_thresh：超过这个阈值，两个anchor就去pk
# anchors：锚框
# batch_size
# config：配置项
# kwargs：其他参数
class proposal(KE.Layer):
    def __init__(self, proposal_count, nms_thresh, anchors, batch_size, config=None, **kwargs):
        super(proposal, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.batch_size = batch_size
        self.config = config
        self.nms_thresh = nms_thresh
    
    # input：输入的 probs置信度inputs[0] 和 deltas修正量inputs[1]
    def call(self, inputs):
        # 取最后一维（前景概率）
        probs = inputs[0][:, :, 1]
        # 修正量
        deltas = inputs[1]
        # -------------------------------- 取得分前100的锚框数据 --------------------------------
        # 乘RPN_BBOX_STD_DEV，是因为build_rpnTarget函数里面除以了这个数，避免值太小
        deltas = deltas*np.reshape(self.config.RPN_BBOX_STD_DEV, (1, 1, 4))
        # 判断：取100和anchors数之间最小
        prenms_num = min(self.anchors.shape[0], 100)
        # 取前100个锚框得分最高的锚框下标
        idxs = tf.nn.top_k(probs, prenms_num).indices

        # 按坐标取元素
        # 置信度
        probs = batch_slice([probs, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        # 修正量
        deltas = batch_slice([deltas, idxs], lambda x,y:tf.gather(x, y), self.batch_size)
        # 锚框
        anchors = batch_slice([idxs], lambda x:tf.gather(self.anchors, x), self.batch_size)

        # ----------------------------------------- 修正 -----------------------------------------
        # 取修正框
        refined_boxes = batch_slice([anchors, deltas], lambda x,y:anchor_refinement(x,y), self.batch_size)
        # 限定坐标的边界，不超过边界，避免修到外面去
        H,W = self.config.image_size[:2]
        windows = np.array([0, 0, H, W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x:boxes_clip(x, windows), self.batch_size)
        
        # 归一化
        normalized_boxes = cliped_boxes / np.array([H, W, H, W])
        # 非极大值抑制方法
        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes, scores, self.proposal_count, self.nms_thresh)
            box = tf.gather(normalized_boxes, idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(normalized_boxes)[0],0)
            box = tf.pad(box, [(0, pad_num), (0,0)])
            return box
        # 进行非极大值抑制切换
        proposals_ = batch_slice([normalized_boxes, probs], nms, self.batch_size)
        return proposals_
    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)
    
    