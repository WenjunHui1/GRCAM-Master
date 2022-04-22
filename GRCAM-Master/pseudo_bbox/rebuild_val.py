import os
import shutil
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='',
                    help='The path for images on ILSVRC2015 validation set.')
parser.add_argument('--rebuild_val_save_path', default='',
                    help='The save path for the rebuild validation set.')
parser.add_argument('--classify_save_path', default='',
                    help='The path for classification result.')
                    
args = parser.parse_args()

for i in range(1000):
    path = os.path.join(args.rebuild_val_save_path, str(i))
    os.mkdir(path)

with open(args.classify_save_path) as f:
    for line in f.readlines():
        image_id, t1, t5, pred  = line.strip('\n').split(',')
        original_path = os.path.join(args.image_path,str(image_id))

        aiming_path = os.path.join(args.rebuild_val_save_path, str(pred))
            
        shutil.copy(original_path, aiming_path)
print('rebuild validation set completed')