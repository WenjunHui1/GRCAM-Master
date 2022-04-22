# -*- coding: utf-8 -*-
__author__ = 'hwj'

import matplotlib
matplotlib.use('Agg')
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from utils_top5_simple import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
import sys
    
class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()    
    
def configure_log():
    log_file_name = os.path.join('train_log','log.log')
    Logger(log_file_name)
    
def five_crops_tf(image):
    image = tf.squeeze(image)
    shape = image.get_shape().as_list()
    assert shape[:2] == [256,256]
    crop_height = 224
    crop_width = 224
    cropped_shape = tf.stack([crop_height, crop_width, shape[2]])
    offsets = [tf.to_int32(tf.stack([0, 0, 0])),
               tf.to_int32(tf.stack([0, 32, 0])),
               tf.to_int32(tf.stack([32, 0, 0])),
               tf.to_int32(tf.stack([32, 32, 0])),
               tf.to_int32(tf.stack([16, 16, 0]))]
    image_list = []
    for offset in offsets:
        image_crop = tf.slice(image, offset, cropped_shape)
        assert image_crop.get_shape().as_list()[:2] == [224,224]
        image_list.append(tf.expand_dims(image_crop,0))
    return tf.concat(image_list,0)

def ten_crops_tf(image):
    image = tf.squeeze(image)
    shape = image.get_shape().as_list()
    assert shape[:2] == [256,256]
    return tf.concat([five_crops_tf(image),five_crops_tf(tf.image.flip_left_right(image)),],0)

def bbox_revise(gt_box, pred_box):
    bbox_gt = tf.constant([gt_box[0][0], gt_box[0][1], gt_box[0][2], gt_box[0][3]], dtype=tf.float32)
    bbox_pred = tf.constant(pred_box, dtype=tf.float32)
    cost_bbox = tf.reduce_mean(tf.square(bbox_gt - bbox_pred)) 
    bbox_grad = tf.gradients(cost_bbox, bbox_pred)
    regress = tf.nn.l2_normalize(tf.nn.sigmoid(bbox_pred),[0]) - tf.nn.l2_normalize(bbox_grad, [0])
    regress = 2 * regress * tf.nn.l2_normalize(bbox_grad, [0])
    bbox_regress = bbox_pred - regress
    bbox_regress = tf.where(bbox_regress < 0, tf.zeros_like(bbox_regress), bbox_regress)
    bbox_regress = tf.where(bbox_regress > 223, tf.ones_like(bbox_regress)*223, bbox_regress)
    return bbox_regress

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_false', help='run offline evaluation instead of training')

    parser.add_argument('--batch_size', default=1, type=int,
                        help="total batch size. "
                        "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy."
                        "Pretrained models listed in README were trained with batch=32x8.")
                        
    parser.add_argument('--image_path', default='',
                        help='The path for images on ILSVRC2015 validation set')
    parser.add_argument('--annotation_path', default='',
                        help='The path for annotations on ILSVRC2015 validation set')
    parser.add_argument('--bounding_box_save_path', default='',
                        help='The save path for the bounding boxes.')
    parser.add_argument('--classify_save_path', default='',
                        help='The save path for the classification results.')
    parser.add_argument('--pseudo_bounding_box_save_path', default='',
                        help='The save path for the pseudo bounding box.')
    args = parser.parse_args()
    batch_size = args.batch_size

    eval_graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    configure_log()

    iou_all = []
    error_all = []
    iou_true = []
    clasify = []
    clasify_top5 = []
    error_all_top5 = []
    try:
        allnum_image = len(os.listdir(args.annotation_path))
        path_annotation = args.annotation_path
        path_img = args.image_path
    except:
        print('data is wrang!!!!!')
        exit(0)

    label_name = [l.split(' ',1)[0] for l in [l.strip() for l in open('synset1.txt').readlines()]]

    th_list = [0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93]
    error1 = {}
    error1_original = {}
    error5 = {}
    error5_original={}
    kernel_list = [(0,0)]
    for i in th_list:
        error1[str(i)] = np.zeros((1,1)) +0.0
        error1_original[str(i)] = np.zeros((1,1)) +0.0
        error5[str(i)] = np.zeros((1,1)) +0.0
        error5_original[str(i)] = np.zeros((1,1)) +0.0
       
    original_bbox = {'ILSVRC2012_val_'+str(i+1).zfill(8)+str('.JPEG'):{th: [] for th in th_list} for i in range(50000)}
    for th in th_list:
        with open(args.bounding_box_save_path+str(th)+'.txt') as f:
            for line in f.readlines():
                image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                original_bbox[image_id][th].append([x0, x1, y0, y1])
                    
    pseudo_boxes = {'ILSVRC2012_val_'+str(i+1).zfill(8)+str('.JPEG'):[] for i in range(50000)}
    with open(args.pseudo_bounding_box_save_path) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            pseudo_boxes[image_id].append([x0, x1, y0, y1])            
    
    top1 = {'ILSVRC2012_val_'+str(i+1).zfill(8)+str('.JPEG'):[] for i in range(50000)}
    top5 = {'ILSVRC2012_val_'+str(i+1).zfill(8)+str('.JPEG'):[] for i in range(50000)}
    with open(args.classify_save_path) as f:
        for line in f.readlines():
            image_id, t1, t5, pred = line.strip('\n').split(',')
            top1[image_id].append(t1)
            top5[image_id].append(t5)
    
    for begin in range(50000):
        tf.reset_default_graph()
        with tf.Session() as sess:
            operations = eval_graph.get_operations()
            collection = eval_graph.get_all_collection_keys()
        
            issave = False
            
            batch_ture_bboxs =  [parse_xml(path_annotation+ labelname)[0] \
                                    for labelname in ['ILSVRC2012_val_000'+str(num_img).zfill(5)+'.xml' \
                                    for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]
            
            batch_image_name = [imagename\
                                    for imagename in ['ILSVRC2012_val_000'+str(num_img).zfill(5) \
                                    for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]        
                        
            batch_image1 = np.array([reshape_orimg(load_image2(path_img + imagename+'.JPEG',batch_ture_bboxs[(int(imagename[-5:])-1)%batch_size])[0]) \
                                        for imagename in batch_image_name])

            for th in th_list:
                
                pred_box = original_bbox[str(batch_image_name[0])+str('.JPEG')][th]
                bbox_regress = bbox_revise(pseudo_boxes[str(batch_image_name[0])+str('.JPEG')], pred_box)
                revise_bbox = bbox_regress.eval(session=sess)
                iou_revise = get_overlaps(batch_ture_bboxs[0], revise_bbox[0])
                iou_original = get_overlaps(batch_ture_bboxs[0], pred_box)
                
                error1[str(th)][0,0] = error1[str(th)][0,0] + int(iou_revise>=0.5) * int(top1[str(batch_image_name[0])+str('.JPEG')][0])
                error1_original[str(th)][0,0] = error1_original[str(th)][0,0] + int(iou_original>=0.5)*int(top1[str(batch_image_name[0])+str('.JPEG')][0])
                error5[str(th)][0,0] = error5[str(th)][0,0] + int(iou_revise>=0.5)*int(top5[str(batch_image_name[0])+str('.JPEG')][0])
                error5_original[str(th)][0,0] = error5_original[str(th)][0,0] + int(iou_original>=0.5)*int(top5[str(batch_image_name[0])+str('.JPEG')][0])
                
            print('######################################################################')
            
            for th in th_list:
                for kernel_1,kernel_2 in kernel_list:
                    
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,' {} th: {}, err1:{},err1_o:{}, err5: {}, err5_o: {}'.format( \
                        begin+1,th,\
                        round(1 - (error1[str(th)][0,0]/((begin+1)*batch_size)),5),
                        round(1 - (error1_original[str(th)][0,0]/((begin+1)*batch_size)),5),
                        round(1 - (error5[str(th)][0,0]/((begin+1)*batch_size)),5),
                        round(1 - (error5_original[str(th)][0,0]/((begin+1)*batch_size)),5),
                        ))
