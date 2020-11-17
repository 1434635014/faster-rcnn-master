from cv2 import cv2 as cv
import os
import tensorflow as tf
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
# from utils import shapeData as dataSet
from config import Config as config

_test = ''
os_path = 'data/voc/'
bbox_path = os_path + 'Annotations'  + _test + '/'
img_path = os_path + 'JPEGImages' + _test + '/'

start_index = 0  # 图片读取开始位置
end_index = 200  # 图片读取结束位置

# 第一个是背景
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
classes_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 按指定图像大小调整尺寸
def resize_image(image, size = config.image_size[0]):
    # 缩放比例
    scale = 0.0
    w_s = 0.0  # 宽度偏移
    h_s = 0.0  # 高度偏移
    # 获取图片尺寸
    h, w, _ = image.shape
    if h > w:
        scale = h / size
        w_s = (size - (w / scale)) / 2
    else:
        scale = w / size
        h_s = (size - (h / scale)) / 2

    top, bottom, left, right = (0,0,0,0)

    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h,w)

    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。

    # RGB颜色
    BLACK = [128,128,128]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv.resize(constant, (size, size)), scale, w_s, h_s

# 计算iou
def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2-y1, 0) * np.maximum(x2-x1, 0)
    union = areas[:] + area - interSec 
    iou = interSec / union
    return iou

# 获取图片中所有的anchor
def anchor_gen(featureMap_size, ratios, scales, rpn_stride, anchor_stride):
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()
    
    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)
    
    shift_x = np.arange(0, featureMap_size[0]) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1]) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)
    
    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes

def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
    areas2 = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:,i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap
def build_rpnTarget(boxes, anchors, index):
    rpn_match = np.zeros(anchors.shape[0],dtype=np.int32)
    rpn_bboxes = np.zeros((config.train_rois_num, 4))
    
    iou = compute_overlap(anchors, boxes)
    maxArg_iou = np.argmax(iou, axis=1)
    max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]
    # 这里要调整成 0.3 ~ 0.7 之间，这里之前是 0.1 ~ 0.4
    postive_anchor_idxs = np.where(max_iou > 0.5)[0]
    negative_anchor_idxs = np.where(max_iou < 0.15)[0]
    
    rpn_match[postive_anchor_idxs]=1
    rpn_match[negative_anchor_idxs]=-1

    # 不好的数据，选个最大的
    maxIou_anchors = np.argmax(iou, axis=0)
    rpn_match[maxIou_anchors] = 1
    
    ids = np.where(rpn_match==1)[0]
    extral = len(ids) - config.train_rois_num // 2
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0
    
    ids = np.where(rpn_match==-1)[0]
    extral = len(ids) - ( config.train_rois_num - np.where(rpn_match==1)[0].shape[0])
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    idxs = np.where(rpn_match==1)[0]
    ix = 0
    
    for i, a in zip(idxs, anchors[idxs]):
        gt = boxes[maxArg_iou[i]]
        
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_centy = gt[0] + 0.5 * gt_h
        gt_centx = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_centy = a[0] + 0.5 * a_h
        a_centx = a[1] + 0.5 * a_w
        
        rpn_bboxes[ix] = [(gt_centy - a_centy)/a_h, (gt_centx - a_centx)/a_w, np.log(gt_h / a_h), np.log(gt_w / a_w)]
        # 归一化，避免偏差太小
        rpn_bboxes[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bboxes, len(idxs), anchors[idxs]

_anchors = anchor_gen(config.featureMap_size, config.ratios, config.scales, config.rpn_stride, config.anchor_stride)

def getAllImage(index):
    img = ''        # 图片  [img, img]
    bbox = []       # 标注框  [[bbox1, bbox2], [bbox1]]
    class_id = []   # 类别  [1, 1]
    active_ids = np.array(classes_arr) # 包含类别所对应的下标

    rpn_match = []
    rpn_bboxes = []
    anchors = []
    to_size = config.image_size[0]   # 统一的尺寸
    
    scale = 0.0     # 变量缩放比例
    w_s = 0.0       # 变量宽度偏移
    h_s = 0.0       # 变量高度偏移

    for root, dirs, files in os.walk(bbox_path):
        DOMTree = xml.dom.minidom.parse(bbox_path + files[index])
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
        img = cv.imread(img_path + filename)
        img, scale, w_s, h_s = resize_image(img, to_size)

        objects = collection.getElementsByTagName("object")
        for obj in objects:
            name = obj.getElementsByTagName("name")[0].childNodes[0].data
            xmin = float(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = float(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = float(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = float(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
            # 两个对角点
            i1_pt1 = (xmin, ymin)
            i1_pt2 = (xmax, ymax)
            bbox.append([int(i1_pt1[0] / scale + w_s), int(i1_pt1[1] / scale + h_s), int(i1_pt2[0] / scale + w_s),  int(i1_pt2[1] / scale + h_s)])
            class_id.append([float(classes.index(name))])
        
        bbox = np.array(bbox)
        class_id = np.array(class_id)

        active_ids_ = np.unique(class_id)
        for k in range(active_ids_.shape[0]):
            active_ids[int(active_ids_[k])]=1

        rpn_match, rpn_bboxes, idxLen, anchors = build_rpnTarget(bbox, _anchors, index)

    return img, bbox, class_id, active_ids, rpn_match, rpn_bboxes, idxLen, anchors


# num_one = 0  # rpn_bboxes大于0的个数
# num_all = 0  # rpn_bboxes等于大于种类的个数
# for img_num in range(200):
#     img, bbox, class_id, active_ids, rpn_match, rpn_bboxes, idxLen, anchors = getAllImage(img_num)
#     if idxLen > 0:
#         num_one += 1
#     if idxLen >= len(class_id):
#         num_all += 1
# print(num_one)
# print(num_all)

# img_num = 4257
# img, bbox, class_id, active_ids, rpn_match, rpn_bboxes, idxLen, anchors = getAllImage(img_num)
# print(len(bbox))
# print(len(active_ids))
# for fi in range(len(bbox)):
#     fold_data = bbox[fi]  # 0：x，1：y，2：w，3：h
#     i1_pt1 = (int(fold_data[0]), int(fold_data[1]))
#     # i1_pt2 = (int(fold_data[0] + fold_data[2]), int(fold_data[1] + fold_data[3]))
#     i1_pt2 = (int(fold_data[2]), int(fold_data[3]))
#     cv.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, color=(255, 0, 255))
# for fi in range(len(anchors)):
#     fold_data = anchors[fi]  # 0：x，1：y，2：w，3：h
#     i1_pt1 = (int(fold_data[0]), int(fold_data[1]))
#     # i1_pt2 = (int(fold_data[0] + fold_data[2]), int(fold_data[1] + fold_data[3]))
#     i1_pt2 = (int(fold_data[2]), int(fold_data[3]))
#     cv.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, color=(255, 0, 255))
# cv.imshow('Image', img)
# cv.waitKey(0)

# for i in range(len(rpn_bboxesList)):
#     for ii in range(len(rpn_bboxesList[i])):
#         isnan = np.isnan(rpn_bboxesList[i][ii])
#         if True in isnan:
#             print('包含NaN，数据：' + str(i))
