import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
import time
import cv2
import argparse

from resnet import *
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

def PCA_svd(feature, center=True):
    components = []
    n = feature.shape[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H, feature)
    
    u, s, v = torch.svd(X_center)
    components.append(torch.unsqueeze(v[0], 0).t())
    components.append(torch.unsqueeze(v[1], 0).t())
    
    return components

def get_configs():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--rebuild_val_save_path', type=str, default='', help='The path for the rebuild validation set.')
    parser.add_argument('--pretrained_path', type=str, default='', help='The path for trained model.')
    parser.add_argument('--pseudo_bounding_box_save_path', type=str, default='', help='The path for saving the pseudo bounding box.')

    args = parser.parse_args()

    return args

class Trainer(object):
 
    def __init__(self):
        self.args = get_configs()
        self.model = self._set_model()

    #建立网络模型
    def _set_model(self):
        GPUID = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
        print("Loading model ...")
        model = resnet50(        
            pretrained_path=self.args.pretrained_path)           
        model = model.cuda()
        return model

    def compute_mean(self):
        rebuild_val_path = self.args.rebuild_val_save_path
        class_name = os.listdir(rebuild_val_path)
        class_number = len(class_name)
        mean_bag = {i:torch.zeros((1,1)) for i in range(class_number)}
        for i in range(class_number):
            if (i+1) % 50 == 0:
                print(i+1,'/',class_number)
            subpath = os.path.join(rebuild_val_path, class_name[i])
            
            file_number = len([x for x in os.listdir(subpath)]) 
            content = os.listdir(subpath)
            
            mean_each = [[],[],[]]
            for j in range(file_number):
                path_image = os.path.join(subpath, content[j])
                image = cv2.imread(path_image)
                
                mean_b = np.mean(image[:, :, 0])
                mean_g = np.mean(image[:, :, 1])
                mean_r = np.mean(image[:, :, 2])
                mean_each[0].append(mean_b)
                mean_each[1].append(mean_g)
                mean_each[2].append(mean_r)
            mean_bag[i] = [np.mean(mean_each[0]), np.mean(mean_each[1]),np.mean(mean_each[2])]
        return mean_bag

    def bbox(self, mean_bag, i):
        num = 0
        rebuild_val_path = self.args.rebuild_val_save_path
        class_name = os.listdir(rebuild_val_path)
        class_number = len([x for x in os.listdir(rebuild_val_path)]) 
        _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
        
        subpath = os.path.join(rebuild_val_path, class_name[i])
        file_number = len([x for x in os.listdir(subpath)])  
        content = os.listdir(subpath)
        images = []
        for j in range(file_number):
            path_image = os.path.join(subpath, content[j])
            image = cv2.imread(path_image)
            image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC) - mean_bag[i]
            image = torch.unsqueeze(torch.Tensor(image).view(3,224,224), 0) 
            images.append(image)
        images = torch.cat(images, 0)
        images = images.cuda()
        with torch.no_grad():
            output = self.model(images)

        feature_map = output['feature_map'].cpu()
        
        feature = feature_map.view(feature_map.shape[0] * feature_map.shape[2] * feature_map.shape[3], -1)
    
        feature_mean = torch.unsqueeze(torch.mean(feature, 0), 0)
        
        new_feature = torch.unsqueeze(feature_mean, 2)
        
        feature = feature - torch.ones_like(feature) * feature_mean
        
        transform_matrix = PCA_svd(feature)
        
        torch.cuda.empty_cache()
        
        for k in range(file_number):
            path_image = os.path.join(subpath, content[k])
            image = cv2.imread(path_image)
            image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC) - mean_bag[i]
            image = torch.Tensor(image).view(3,224,224)
            image = torch.unsqueeze(image, 0).cuda() 
            with torch.no_grad():
                output = self.model(image, 1)
            feature_map = output['feature_map'].cpu()
            feature1 = torch.squeeze(feature_map) - torch.ones((2048,28,28)) * torch.unsqueeze(torch.squeeze(new_feature,0), 1)
            feature_mean = torch.zeros(28, 28)
            for h in range(feature1.shape[1]):
                for w in range(feature1.shape[2]):    
                    feature_mean[h][w] = torch.mean(torch.mm(torch.unsqueeze(feature1[:, h, w], 0), torch.cat(transform_matrix,1)), 1)
            feature_sum = feature_mean.detach().cpu().numpy().astype(np.float)
            
            cam_resized = cv2.resize(feature_sum, (224, 224),interpolation=cv2.INTER_CUBIC)
            scoremap = np.expand_dims(cam_resized.astype(np.uint8), 2)
            width, height = cam_resized.shape
            _, thr_gray_heatmap = cv2.threshold(         
                    src=scoremap,
                    thresh=0,
                    maxval=255,
                    type=cv2.THRESH_BINARY)
            contours = cv2.findContours(image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
            if len(contours)==0:
                coordinate = [0, 0, 0, 0]
            else:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                x0, y0, x1, y1 = x, y, x + w, y + h
                x1 = min(x1, width - 1)
                y1 = min(y1, height - 1)
                
                coordinate = [x0,x1,y0,y1]
            
            with open(self.args.pseudo_bounding_box_save_path, 'a') as f:
                f.write(str(content[k])+str(',')+str(coordinate).strip('[').strip(' ').strip(']')+'\n')
            
def main():
    trainer = Trainer()
    print('####################################################')
    print("unsupervision object localization begining ...") 
    
    mean_bag = trainer.compute_mean()
    for i in range(1000):
        start = time.time()
       
        trainer.bbox(mean_bag, i)
        end = time.time()
        print(i,(end-start)/60,'min')
        

if __name__ == '__main__':
    main()
 
