#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-21 17:52:14
lastTime     : 2024-10-22 09:06:03
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/voc2yolo.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw,ImageFont
import xml.etree.ElementTree as ET
import shutil
import math
import cv2

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id, dataset_path):
    
    # classes = ['cup', 'can','fruitpeel','box','bottle','liquid']
    # classes = ['cup', 'can','fruitpeel','box','bottle','liquid','peanut','cigarette']
    # classes = ['peanut', 'cigarette', 'liquid']
    # classes = ['Drain Hole', 'Pothole','Sewer Cover']
    classes = ["person",
                "rider",
                "car",
                "bus",
                "truck",
                "bike",
                "motor",
                "traffic light", 
                "traffic sign",
                "train"]
    print(image_id)
    image_id = image_id.split('.png')[0]
    # image_id = ["%s_" % tmp for tmp in image_id.split('.')[:-1]]
    
    outpput_path = os.path.join(dataset_path,'txt_Annotations')
    if not os.path.exists(outpput_path):
        os.makedirs(outpput_path)    
    
    try:
        in_file = open(dataset_path + '/annotations/%s.xml'%(image_id))
    

        out_file = open(outpput_path + '/%s.txt'%(image_id),'w') #生成txt格式文件
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')  
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes :
                print("cls: ",cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')   
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except:
        out_file = open(outpput_path + '/%s.txt'%(image_id),'w') #生成txt格式文件

def convert_annotation_voc2yolo(dataset_path):
    
    file_list = open(os.path.join(dataset_path,'test.txt'),'w')
    image_path = dataset_path + '/images'
    index_img = os.listdir(image_path)
    for id_img in index_img:
        file_list.write('%s\n'%(id_img))
    file_list.close()
    
    image_ids_train = open(os.path.join(dataset_path,'test.txt')).read().strip().split()
    list_file_train = open('save_test.txt', 'w')     
    for image_id in image_ids_train:
        list_file_train.write('./dataset-yolo/%s\n'%(image_id))
        print(image_id)  
        convert_annotation(image_id.split('.jpg')[0], dataset_path)
