# -*- coding: utf-8 -*-
__author__ = 'hwj'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import matplotlib
matplotlib.use('Agg')
import argparse
import inceptionv3
from utils_top5_simple import *
from tensorpack.models import *
from tensorpack.tfutils import argscope, get_model_loader
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from imagenet_utils import ImageNetModel
from resnet_model import (
    preresnet_basicblock, preresnet_bottleneck, preresnet_group,
    resnet_backbone, resnet_group,
    resnet_basicblock, resnet_bottleneck, resnext_32x4d_bottleneck, se_resnet_bottleneck)
from tensorpack.tfutils.tower import PredictTowerContext

import tf_slim as slim

import tensorflow.compat.v1 as tf
import vgg16
import sys

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0 
FCweights_name = {"cub_inceptionv3": 'InceptionV3/Logits/Conv2d_1c_1x1_200/weights:0',
           "cub_resnet50se": 'linearcub/W:0', 
           "cub_vgg": 'linear_cub/W:0', 
           "imagenet_inceptionv3": 'InceptionV3/Logits/Conv2d_1c_1x1_1000/weights:0', 
           "imagenet_resnet50se": 'linear/W:0', 
           "imagenet_vgg": 'linear/W:0'}
def scoremap2bbox(scoremap_image,threshold):
    height, width = scoremap_image.shape
    scoremap_image = scoremap_image - scoremap_image.min()
    scoremap_image = scoremap_image / scoremap_image.max()
    scoremap_image = np.expand_dims((scoremap_image * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, x1, y0, y1])

    return np.asarray(estimated_boxes), len(contours)
 
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


class Model(ImageNetModel):
    def __init__(self, depth, mode='resnet50se'):

        self.mode = mode
        self.classnum = 1000

    def get_logits(self, image):
        if self.mode in ['vgg','resnet50se']:
            with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
                with PredictTowerContext(''):
                    return vgg16.vgg_gap(image, self.classnum) if self.mode == 'vgg' else resnet_backbone(image, [3, 4, 6, 3],resnet_group, se_resnet_bottleneck)
        else: 
            is_training = False
            with slim.arg_scope(inceptionv3.inception_v3_arg_scope(weight_decay=0.0005)):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}):
                    return inceptionv3.inception_v3(image, 1000,is_training=is_training,global_pool=True)#

def get_1maxarg(matrix):
    matrix_max = tf.zeros_like(matrix) + tf.reduce_max(matrix,axis=1,keep_dims=True)
    matrix1 = tf.where(matrix < matrix_max, tf.zeros_like(matrix), matrix)
    
    return tf.maximum(tf.sign(matrix1) ,0.0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load', default='',help='load a model for training or evaluation')
    parser.add_argument('--data-format', help='the image data layout used by the model',
                        default='NHWC', choices=['NCHW', 'NHWC'])
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=50, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--weight-decay-norm', action='store_true',
                        help="apply weight decay on normalization layers (gamma & beta)."
                             "This is used in torch/pytorch, and slightly "
                             "improves validation accuracy of large models.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="total batch size. "
                        "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy."
                        "Pretrained models listed in README were trained with batch=32x8.")
    parser.add_argument('--mode', choices=['resnet50se', 'vgg', 'inceptionv3'],
                        help='variants of resnet to use', default='resnet50se')
    parser.add_argument('--image_path', default='',
                        help='The path for images on ILSVRC2015 validation set')
    parser.add_argument('--annotation_path', default='',
                        help='The path for annotations on ILSVRC2015 validation set')
    parser.add_argument('--bounding_box_save_path', default='',
                        help='The save path for the bounding boxes.')
    parser.add_argument('--classify_save_path', default='',
                        help='The save path for the classification results.')
                        
    args = parser.parse_args()
    batch_size = args.batch_size

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        
        timages = tf.placeholder(tf.string, [batch_size, ])
        tlabels = tf.placeholder(tf.int32, [batch_size, ])
        
        model = Model(args.depth, args.mode)
        batch_image_graph = []
        batch_image_graph_10crops = []
        images1_split = tf.split(timages, batch_size, 0)
        for image_someone in images1_split:
            image_size = 224
            image = tf.image.resize_bilinear(tf.expand_dims(tf.image.decode_jpeg(tf.read_file(tf.squeeze(image_someone)), channels=3),0), [224, 224],align_corners=False)
            image_someone_pre = model.image_preprocess(tf.to_float(tf.reverse(image,axis=[-1])))

            image_10crops = ten_crops_tf(tf.image.resize_bilinear(tf.expand_dims(tf.image.decode_jpeg(tf.read_file(tf.squeeze(image_someone)), channels=3),0), [256, 256],align_corners=False))
            image_someone_pre_10crops = model.image_preprocess(tf.to_float(tf.reverse(image_10crops,axis=[-1])))


            batch_image_graph_10crops.append(image_someone_pre_10crops)
            batch_image_graph.append(image_someone_pre)
        preprocessed_images = tf.concat(batch_image_graph,0)
        preprocessed_images_10crops = tf.concat(batch_image_graph_10crops,0)
        
        model.data_format = args.data_format
        if args.weight_decay_norm:
            model.weight_decay_pattern = ".*/W|.*/gamma|.*/beta"
        logits,target_conv_layer = model.get_logits(preprocessed_images)
        with tf.variable_scope('', reuse=True):
            preprocessed_images_10crops_return = model.get_logits(preprocessed_images_10crops)
        prob_10crops = tf.nn.softmax(preprocessed_images_10crops_return[0])
        prob_10crops_ave = tf.expand_dims(tf.reduce_mean(prob_10crops,0),0)

        intopk = tf.nn.in_top_k(prob_10crops_ave, tlabels, 1)
        intopk5 = tf.nn.in_top_k(prob_10crops_ave, tlabels, 5)
        argmax_prob = tf.argmax(prob_10crops_ave,axis =1)

        cost_gt = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=10*tf.one_hot(tlabels,1000))
        
        target_conv_layer_grad_gt =tf.stop_gradient(tf.gradients(cost_gt, target_conv_layer)[0])
        mask_gt = tf.nn.l2_normalize(tf.nn.sigmoid(target_conv_layer),[1,2,3]) - tf.nn.l2_normalize(target_conv_layer_grad_gt,[1,2,3])#map_attention放到l2_normalize外面试一下
        experiment_layer_grad_gt = target_conv_layer * mask_gt
      
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    configure_log()
    with tf.Session(graph=eval_graph,config=config) as sess:
        with tf.device("/gpu:0"):
            latest_checkpoint = args.load
            try:
                saver.restore(sess,latest_checkpoint)
                print('**************** reload success *********************',latest_checkpoint)
            except:
                print('**************** reload failed *********************',latest_checkpoint)
                exit(0)

            operations = eval_graph.get_operations()
            collection = eval_graph.get_all_collection_keys()
            
            weights_cam = sess.run(tf.squeeze(eval_graph.get_tensor_by_name(FCweights_name["imagenet_resnet50se"]))).T
        
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
            
            th_list = [0.85, 0.86, 0.87,0.88,0.89,0.90,0.91,0.92,0.93]

            top1 = 0.0
            top5 = 0.0

            for begin in range(50000):
                print(begin)
                issave = False
                
                batch_ture_bboxs =  [parse_xml(path_annotation+ labelname)[0] \
                                        for labelname in ['ILSVRC2012_val_000'+str(num_img).zfill(5)+'.xml' \
                                        for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]
                                        
                batch_num_label  =  np.array([label_name.index(parse_xml(path_annotation+ labelname)[1]) \
                                        for labelname in ['ILSVRC2012_val_000'+str(num_img).zfill(5)+'.xml' \
                                        for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]).astype(np.int32).reshape([batch_size,])
                
                batch_image_name = [imagename\
                                        for imagename in ['ILSVRC2012_val_000'+str(num_img).zfill(5) \
                                        for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]        
                batch_image_name1 = [path_img + imagename+'.JPEG'\
                                        for imagename in ['ILSVRC2012_val_000'+str(num_img).zfill(5) \
                                        for num_img in range(begin* batch_size + 1, (begin + 1)* batch_size + 1)]]
                            
                            
                batch_image1 = np.array([reshape_orimg(load_image2(path_img + imagename+'.JPEG',batch_ture_bboxs[(int(imagename[-5:])-1)%batch_size])[0]) \
                                            for imagename in batch_image_name])

                theintopk ,theintopk5, argmax_prob_run, up_max_pool5_batch_gt   = \
                                                 sess.run([intopk, intopk5, argmax_prob, experiment_layer_grad_gt], \
                                                 feed_dict={timages: batch_image_name1, tlabels: batch_num_label})
                
                with open(args.classify_save_path, 'a') as f:       
                    t1 = 1 if theintopk else 0
                    t5 = 1 if theintopk5 else 0
                    pred =  argmax_prob_run[0]
                    f.write(str(batch_image_name[0])+str('.JPEG')+','+str(t1)+','+str(t5)+','+str(pred)+str('\n'))
                
                for batch_size_num in range(batch_size):
                    argmax_prob_run = batch_num_label[0]
                    shape = up_max_pool5_batch_gt[batch_size_num,:,:,:].shape[0:2]
                    
                    reshape_channel = 2048 if args.mode == 'resnet50se' else 1024
                    reshape_size = 28 if args.mode in ['inceptionv3','resnet50se'] else 14
                    cam_gt = np.dot(weights_cam[argmax_prob_run,:].reshape((1,reshape_channel)),up_max_pool5_batch_gt[batch_size_num,:,:,:].transpose(2,0,1).reshape(reshape_channel,-1)).reshape(shape)
                    
                    cam_gt = cam_gt - cam_gt.min()
                    up_max_pool5_run_gt = cam_gt / cam_gt.max()
                    
                    dot_gt = cv2.resize(up_max_pool5_run_gt , (224, 224))
                    for th in th_list: 
                        preboxxs, preboxxs_num = scoremap2bbox(dot_gt,1 - th)
                        x_min, x_max, y_min, y_max = preboxxs[0]
                        iou_gt = get_overlaps(batch_ture_bboxs[batch_size_num], [[x_min,x_max,y_min,y_max]])
                        cooridinate = [x_min, x_max, y_min, y_max]
                        
                        with open(args.bounding_box_save_path+str(round(th, 3))+str('.txt'), 'a') as f:
                            f.write(str(batch_image_name[0])+str('.JPEG')+','+str(cooridinate).strip('[').strip(']')+str('\n'))