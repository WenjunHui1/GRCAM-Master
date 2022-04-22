# -*- coding: utf-8 -*-
__author__ = 'hwj'

import skimage
import skimage.io
import skimage.transform
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt
from xml.dom.minidom import parse
from skimage import measure


def reshape_orimg(image):
    w,h = image.shape[0:2]
    try:
        image = image.reshape((w,h,3))
    except:
        image = np.tile(np.expand_dims(image,2),[1,1,3]).reshape((w,h,3))
    return image
def get_overlaps(bb_true, bb_pre):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    # bb = [x_min, x_max, y_min, y_max]
    Parameters
    ----------
    bb1 : dict
        Keys: {x1, x2, y1, y2}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {x1, x2, y1, y2}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # bb2 = bb_pre
    iou_ = []
    # print('type(bb2): ', type(bb2), len(bb2))
    # print('type(bb_true): ', type(bb_true),len(bb_true))
    for bb2 in bb_pre:
        for bb1 in bb_true:
            iou = 0.0
            # bb1 = bb_true[num]
            # print(num, ' type(bb1): ', type(bb1), len(bb1))
            if bb1[0] >= bb1[1] or bb1[2] >= bb1[3] or bb2[0] >= bb2[1] or bb2[2] >= bb2[3]:
                iou_.append(0.0)
                continue
            # determine the coordinates of the intersection rectangle
            x_left = max(bb1[0], bb2[0])
            y_top = max(bb1[2], bb2[2])
            x_right = min(bb1[1], bb2[1])
            y_bottom = min(bb1[3], bb2[3])

            if x_right < x_left or y_bottom < y_top:
                iou_.append(0.0)
                continue
            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

            # compute the area of both AABBs
            bb1_area = (bb1[1] - bb1[0] + 1) * (bb1[3] - bb1[2] + 1)
            bb2_area = (bb2[1] - bb2[0] + 1) * (bb2[3] - bb2[2] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            if iou > 1.0 or iou < 0.0:
                iou_.append(0.0)
                continue
            iou_.append(iou)
    # print(iou_)
    return max(iou_)
def mak_dir(dir):
    if not os.path.exists('./' + dir):
        os.makedirs("./" + dir)
def print_mes(dot,name = 'array'):
    print(name + '.shape: ',dot.shape)
    print(name + '.max: ', dot.max())
    print(name + '.min: ', dot.min())
    print(name + '.sum: ', dot.sum())
    print(name + '.sum_abs: ', np.abs(dot).sum())
    print(name + ' no equal to 0', (dot != 0).sum() )
    print(name + ' more than 0', (dot > 0).sum()  )
    print(name + ' less than 0', (dot < 0).sum()  )
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def plt_bboxs(bboxs,color = 'r'):
    for num in range(len(bboxs)):
        assert len(bboxs[num]) == 4, 'bboxs''s type is wrong'
        x_min, x_max, y_min, y_max = bboxs[num]
        plt.plot(x_min, y_min,'r*')
        plt.plot(x_min, y_max,'r*')
        plt.plot(x_max, y_min,'r*')
        plt.plot(x_max, y_max,'r*')

        plt.plot([x_min,x_min], [y_min, y_max],color)
        plt.plot([x_max,x_max], [y_min, y_max],color)
        plt.plot([x_min, x_max],[y_min,y_min], color)
        plt.plot([x_min, x_max],[y_max,y_max], color)


def parse_xml(fn):
    xml_file = parse(fn)
    eles = xml_file.documentElement
    # print(eles.tagName)

    xmin = [int(i.firstChild.data) for i in eles.getElementsByTagName("xmin")]
    xmax = [int(i.firstChild.data) for i in eles.getElementsByTagName("xmax")]
    ymin = [int(i.firstChild.data) for i in eles.getElementsByTagName("ymin")]
    ymax = [int(i.firstChild.data) for i in eles.getElementsByTagName("ymax")]
    name = list(set([i.firstChild.data for i in eles.getElementsByTagName("name")]))
    width = int(eles.getElementsByTagName("width")[0].firstChild.data)
    height = int(eles.getElementsByTagName("height")[0].firstChild.data)
    depth = int(eles.getElementsByTagName("depth")[0].firstChild.data)
    # print(xmin, xmax, ymin, ymax, width, height, depth)
    # return int(xmin), int(xmax), int(ymin), int(ymax), int(width), int(height), int(depth)
    bboxs = []
    for i in range(len(xmin)):
        bboxs.append([xmin[i], xmax[i], ymin[i], ymax[i]])
    # print(bboxs)
    assert len(name) == 1, 'img label not one'
    return bboxs, name[0], width, height, depth
def load_image2(path, bboxs = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],size = 224.0):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    # print('img.shape : ',img.shape)

    assert (0 <= img).all() and (img <= 1.0).all()
    x_ = img.shape[0] + 0.0
    y_ = img.shape[1] + 0.0
    # print('x_,y_: ',x_,y_)
    resized_img = skimage.transform.resize(img, (size, size), preserve_range=True)
    x_scale = ( size / x_) + 0.0
    y_scale = ( size / y_ ) + 0.0
    # print('old bboxs: ', bboxs)
    for num in range(len(bboxs)):
        xmin = int(np.round(bboxs[num][0] * y_scale))
        xmax = int(np.round(bboxs[num][1] * y_scale))
        ymin = int(np.round(bboxs[num][2] * x_scale))
        ymax = int(np.round(bboxs[num][3] * x_scale))
        bboxs[num] = [xmin, xmax, ymin, ymax]
    # print('new bboxs: ', bboxs)
    return resized_img, bboxs
    
def load_image_keepratio(path, bboxs = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]],resize_side_min = 224):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    # print("Original Image Shape: ", img.shape)

    assert (0 <= img).all() and (img <= 1.0).all()
    x_ = img.shape[0]
    y_ = img.shape[1]
    short_edge = min([x_,y_])
    scale = resize_side_min / short_edge
    newx,newy = int(round(x_*scale)),int(round(y_*scale))
    
    # print('x_,y_: ',x_,y_)
    # resized_img = skimage.transform.resize(img, (newx,newy), preserve_range=True)
    resized_img = cv2.resize(img, (newy,newx) )
    x_scale = scale
    y_scale = scale
    # print('old bboxs: ', bboxs)
    for num in range(len(bboxs)):
        xmin = int(np.round(bboxs[num][0] * y_scale))
        xmax = int(np.round(bboxs[num][1] * y_scale))
        ymin = int(np.round(bboxs[num][2] * x_scale))
        ymax = int(np.round(bboxs[num][3] * x_scale))
        bboxs[num] = [xmin, xmax, ymin, ymax]
    # print('new bboxs: ', bboxs)
    # print('resized_img.shape : ',resized_img.shape)
    assert newx == resize_side_min or newy == resize_side_min
    return resized_img, bboxs
def load_image(path, bboxs = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    # print('img.shape : ',img.shape)
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    # print('img.shape[:2] : ',img.shape[:2],'short_edge : ',short_edge)
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    # print('old bboxs: ', bboxs)
    for num in range(len(bboxs)):
        bboxs[num][0] = bboxs[num][0] - xx
        bboxs[num][1] = bboxs[num][1] - xx
        bboxs[num][2] = bboxs[num][2] - yy
        bboxs[num][3] = bboxs[num][3] - yy
        if bboxs[num][0]<0         : bboxs[num][0] = 0
        if bboxs[num][1]>short_edge: bboxs[num][1] = short_edge
        if bboxs[num][1]<0         : bboxs[num][1] = 0
        if bboxs[num][2]<0         : bboxs[num][2] = 0
        if bboxs[num][3]>short_edge: bboxs[num][3] = short_edge
        if bboxs[num][3]<0         : bboxs[num][3] = 0
        bboxs[num] = [int(i * 224/short_edge) for i in bboxs[num]]
        # print('new bboxs: ', bboxs)
    # print('xx,yy: ',xx,yy)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    # print('crop_img.shape : ',crop_img.shape)
    resized_img = skimage.transform.resize(crop_img, (224, 224), preserve_range=True)
    return resized_img, bboxs



def to_heatmap(map):
    map = np.array(map)
    if len(map.shape) == 2: map = np.expand_dims(map,0)
    cmap = np.ones([map.shape[1], map.shape[2], 3,])
    for i in range(map.shape[1]):
        for j in range(map.shape[2]):
            value = map[0,i,j,]
            if value<= 0.25:
                cmap[i,j,0,] = 0
                cmap[i,j,1,] = 4*value
            elif value <= 0.5:
                cmap[i,j,0,] = 0
                cmap[i,j,2,] = 2 - 4*value
            elif value <= 0.75:
                cmap[i,j,0,] = 0
                cmap[i,j,2,] = 4*value - 2
            else :
                cmap[i,j,1,] = 4 - 4*value
                cmap[i,j,2,] = 0
    return cmap

def print_prob(prob, file_path):
    prob = prob.reshape((1000,))
    synset = np.array([l.strip() for l in open(file_path).readlines()])

    # print prob
    pred = np.argsort(prob)[::-1]
    print("pred.shape: ",pred.shape,'   ',pred[0:7])
    # print(pred)
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1
    
    
def removefile(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  # 取文件路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):  
                path_file2 =os.path.join(path_file,f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

  
  



