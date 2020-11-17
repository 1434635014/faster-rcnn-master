from cv2 import cv2 as cv
import os
import tensorflow as tf
import numpy as np
# from utils import shapeData as dataSet
from config import Config as config

_test = ''
os_path = 'data/celebA/'
bbox_path = os_path + 'list_bbox_celeba' + _test + '.txt'
img_path = os_path + 'img' + _test + '/'

start_index = 0  # 图片读取开始位置
end_index = 200  # 图片读取结束位置

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
    postive_anchor_idxs = np.where(max_iou > 0.7)[0]
    negative_anchor_idxs = np.where(max_iou < 0.3)[0]
    
    rpn_match[postive_anchor_idxs]=1
    rpn_match[negative_anchor_idxs]=-1

    # 不好的数据，选个最大的
    max_value = iou[np.argmax(iou)]
    if max_value > 0.5 and max_value <= 0.7:
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

    
def getAllImage():
    imgList = []        # 图片  [img, img]
    bboxList = []       # 标注框  [[bbox1, bbox2], [bbox1]]
    class_idList = []   # 类别  [1, 1]

    rpn_matchList = []
    rpn_bboxesList = []
    anchorsList = []
    to_size = config.image_size[0]   # 统一的尺寸
    scale = 0.0     # 变量缩放比例
    w_s = 0.0       # 变量宽度偏移
    h_s = 0.0       # 变量高度偏移
    num = 0         # 读取的图片个数
    with open(bbox_path, 'r') as f:
        line = f.readline().strip()

        while line:
            if num >= start_index and num < end_index:
                fold_data = line.split()
                img = cv.imread(img_path + fold_data[0])
                img, scale, w_s, h_s = resize_image(img, to_size)
                # 两个对角点
                i1_pt1 = (float(fold_data[1]), float(fold_data[2]))
                i1_pt2 = (float(fold_data[1]) + float(fold_data[3]), float(fold_data[2]) + float(fold_data[4]))
                bbox = np.array([[int(i1_pt1[0] / scale + w_s), int(i1_pt1[1] / scale + h_s), int(i1_pt2[0] / scale + w_s),  int(i1_pt2[1] / scale + h_s)]])
                class_id = np.array([[1]])
                rpn_match, rpn_bboxes, idxLen, anchors = build_rpnTarget(bbox, _anchors, num)

                if num % 100 == 0:
                    print('正在已读取' + str(num) + '个......')
                    
                # 检测rpn_boxs个数是否大于1
                if idxLen > 0 or _test != '':
                    imgList.append(img)
                    bboxList.append(bbox)
                    class_idList.append(class_id)
                    rpn_matchList.append(rpn_match)
                    rpn_bboxesList.append(rpn_bboxes)
                    anchorsList.append(anchors)
                else:
                    print(fold_data[0])
            if num > end_index:
                break
            # if num % 111 == 0 and num != 0:
            num = num + 1
            line = f.readline().strip()

    return imgList, bboxList, class_idList, rpn_matchList, rpn_bboxesList, len(rpn_bboxesList), anchorsList


# imgList, bboxList, class_idList, rpn_matchList, rpn_bboxesList, maxTwoNum, anchorsList = getAllImage()
# img_num = 0
# print(len(imgList))
# img = imgList[img_num]
# class_id = class_idList[img_num]
# for fi in range(len(bboxList[img_num])):
#     fold_data = bboxList[img_num][fi]  # 0：x，1：y，2：w，3：h
#     i1_pt1 = (int(fold_data[0]), int(fold_data[1]))
#     # i1_pt2 = (int(fold_data[0] + fold_data[2]), int(fold_data[1] + fold_data[3]))
#     i1_pt2 = (int(fold_data[2]), int(fold_data[3]))
#     cv.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, color=(255, 0, 255))
# for fi in range(len(anchorsList[img_num])):
#     fold_data = anchorsList[img_num][fi]  # 0：x，1：y，2：w，3：h
#     i1_pt1 = (int(fold_data[0]), int(fold_data[1]))
#     # i1_pt2 = (int(fold_data[0] + fold_data[2]), int(fold_data[1] + fold_data[3]))
#     i1_pt2 = (int(fold_data[2]), int(fold_data[3]))
#     cv.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, color=(255, 0, 255))
# cv.imshow('Image', img)
# cv.waitKey(0)
# print(maxTwoNum)
# print(maxTwoNum / (end_index - start_index))

# for i in range(len(rpn_bboxesList)):
#     for ii in range(len(rpn_bboxesList[i])):
#         isnan = np.isnan(rpn_bboxesList[i][ii])
#         if True in isnan:
#             print('包含NaN，数据：' + str(i))
