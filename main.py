#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-21 15:25:03
lastTime     : 2024-10-22 08:54:45
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/main.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-05-13 08:52:40
lastTime     : 2024-06-27 13:43:05
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /data_process_project/main.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

# -*- coding: utf8 -*-
import os
import json

import xml.etree.ElementTree as xmlET
import xml.etree.ElementTree as ET

import argparse
import shutil
import cv2
import random
import numpy as np

from label2det_v1 import change_dir
from bdd2coco import bdd2coco_detection
from coco2voc import mkr,get_CK5
from PIL import Image, ImageDraw,ImageFont
from dataset_split import dataset_split
from voc2yolo import convert_annotation_voc2yolo

color_list = [(0, 0, 0),(100, 149, 237), (0, 0, 255), (173, 255, 47), (240, 255, 255), (0, 100, 0),
              (47, 79, 79), (255, 228, 196), (138, 43, 226), (165, 42, 42), (222, 184, 135)]

def show_voc_img(dataset_path):
    # classes = ('__background__', # always index 0
    #         'person', 'trafficcone','stairs','escalator','guardrail')
    # classes = ('__background__', # always index 0
    #         'cup', 'can','fruitpeel','box','bottle','liquid','peanut','cigarette')
    classes = ("__background__", # always index 0
                "person",
                "rider",
                "car",
                "bus",
                "truck",
                "bike",
                "motor",
                "traffic light", 
                "traffic sign",
                "train")

    file_path_img =  dataset_path + '/images'
    file_path_xml =  dataset_path + '/annotations'
    save_file_path = dataset_path + '/vis_jpeg'

    # file_path_img =  '/home/oem/pgf/ai_model_train/yolov7_pipeline/yolov7/runs/detect/source_img/0'
    # file_path_xml =  '/home/oem/pgf/ai_model_train/yolov7_pipeline/yolov7/runs/detect/exp/voc_labels/0'
    # save_file_path = '/home/oem/pgf/ai_model_train/yolov7_pipeline/yolov7/runs/detect/vis_jpeg'
    
    pathDir = os.listdir(file_path_img)
    for idx in range(len(pathDir)):  
    # for idx in range(5):    
        filename = pathDir[idx]
        
        try:
            tree = xmlET.parse(os.path.join(file_path_xml, filename.replace('.jpg','.xml')))
        except:
            print(filename)
        
        objs = tree.findall('object')        
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 5), dtype=np.uint16)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # x1 = float(bbox.find('xmin').text) - 1
            # y1 = float(bbox.find('ymin').text) - 1
            # x2 = float(bbox.find('xmax').text) - 1
            # y2 = float(bbox.find('ymax').text) - 1

            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            cla = obj.find('name').text 
            if(cla == "__background__"):
                print(filename)
                print(cla)
            print(cla)
            print(filename)
            print(cla,filename)
            label = classes.index(cla)

            boxes[ix, 0:4] = [x1, y1, x2, y2]
            boxes[ix, 4] = label

        image_name = os.path.splitext(filename)[0]
        # print(image_name)
        try:
            img = Image.open(os.path.join(file_path_img, image_name + '.jpg')) 
        except:
            print(image_name)
            continue

        draw = ImageDraw.Draw(img)
        for ix in range(len(boxes)):
            xmin = int(boxes[ix, 0])
            ymin = int(boxes[ix, 1])
            xmax = int(boxes[ix, 2])
            ymax = int(boxes[ix, 3])
            cls_id = int(boxes[ix, 4])
            font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
            
            try:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color_list[cls_id])
                draw.text([xmin, ymin], classes[boxes[ix, 4]],color_list[cls_id],font=font)
            except:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=140)
                draw.text([xmin, ymin], classes[boxes[ix, 4]],140,font=font)        
            

        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        # cv2.imwrite(os.path.join(save_file_path, image_name + '.png'),img)
        img.save(os.path.join(save_file_path, image_name + '.jpg'))

def bdd2coco(srcDir, dataType, task):

    #step 1: 将json合并
    merge_json_path = os.path.join(srcDir,task,'save_json')
    os.makedirs(merge_json_path,exist_ok=True)
    label_dir = os.path.join(srcDir,'labels','100k')
    change_dir(label_dir, merge_json_path,dataType)

    #step 2: 将合并后的json转换成coco格式
    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bike"},
        {"supercategory": "none", "id": 7, "name": "motor"},
        {"supercategory": "none", "id": 8, "name": "traffic light"},
        {"supercategory": "none", "id": 9, "name": "traffic sign"},
        {"supercategory": "none", "id": 10, "name": "train"}
    ]
    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    for data_type in dataType:
        # create BDD training set detections in COCO format
        print('Loading training set...')
        with open(os.path.join(merge_json_path,
                            'bdd100k_labels_images_{}.json'.format(data_type))) as f:
            train_labels = json.load(f)
        print('Converting {} set...'.format(data_type))

        out_fn = os.path.join(merge_json_path,
                            'bdd100k_labels_images_det_coco_{}.json'.format(data_type))
        bdd2coco_detection(attr_id_dict, train_labels, out_fn)
    
def coco2voc(srcDir, dataTypes, task):
    base_dir = os.path.join(srcDir, task, 'coco2024_voc')
    save_image_dir = os.path.join(base_dir, 'images')  # 在上述文件夹中生成images，annotations两个子文件夹
    save_anno_dir = os.path.join(base_dir, 'annotations')
    mkr(save_image_dir)
    mkr(save_anno_dir)

    origin_image_dir = os.path.join(srcDir, 'images')
    verbose = False  # 是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    origin_anno_dir = os.path.join(srcDir,task,'save_json')
    save_path = [save_image_dir, save_anno_dir]
    get_CK5(origin_anno_dir, origin_image_dir, save_path,verbose)

    return base_dir

def object_convert_main(srcDir,task):
    dataTypes = ['train', 'val']
    
    # #step 1: bdd100k2coco 
    # bdd2coco(srcDir, dataTypes, task) 

    # # step 2: coco2voc
    voc_root = coco2voc(srcDir, dataTypes, task)      

    # #step 2: label visualization
    # show_voc_img(voc_root)   #可视化

    #step 3: voc2yolo
    convert_annotation_voc2yolo(voc_root)

    #step 4: dataset split and yolo format
    dataset_split(voc_root)


def lane_convert_main(srcDir,dstDir,srcjpgfolder):
    
    lane_classes = ('__background__', # always index 0
            'road curb', 'white', 'yellow')
    lane_attri = ['Single','Double','Full','Dashed']

    lane_direct = ['Parallel', 'Vertical']

    lane_info = []
    lane_info.append(lane_classes)
    lane_info.append(lane_attri)
    lane_info.append(lane_direct)


    #step 1: scalabel lane annotation -> pascal xml
    i = 1
    for dirpath, dirnames, filenames in os.walk(srcDir):
        for filepath in filenames:
            fileName = os.path.join(dirpath,filepath)
            print(fileName)
            print("processing: {}, {}".format(i, fileName))
            i = i + 1
            xmlFileName = filepath[:-5] # remove ".json" 5 character
            # 解析该json文件，返回一个列表的列表，存储了一个json文件里面的所有方框坐标及其所属的类
            f = open(fileName)
            info = json.load(f)
            sam_list = info['frames']
            print('annotation sample num: ', len(sam_list))

            #创建修改json文件夹
            os.makedirs('./auto_edit_json/json',exist_ok=True)
            fix_name = './auto_edit_json/json/' + fileName.split('/')[-1]
            f1 = open(fix_name,'a')
            f1.write('[')

            save_fix_img_path = './auto_edit_json/img'
            os.makedirs(save_fix_img_path,exist_ok=True)            

            for sample in sam_list:                                  
                objs = parseJson.parse_lane_Json(sample,lane_info) 
                xml_name = sample['name'].split('/')[-1]
                labels = sample['labels']

                # 如果存在检测对象，创建一个与该json文件具有相同名的VOC格式的xml文件
                if len(objs):
                    xml_voc_lane_writer(xml_name,dstDir,objs,1280,512, srcjpgfolder)                        
                else:
                    print(fileName)
                    continue
            
            for sample in sam_list:                                  
                objs = parseJson.parse_lane_Json(sample,lane_info) 
                xml_name = sample['name'].split('/')[-1]
                labels = sample['labels'] 
                src_img_path = sample['name'].replace('/items/dataset','.') 
                img_fix_name = sample['name'].replace('origin/','auto_edit_json/img/') 
                img_fix_name = img_fix_name.replace('/items/dataset','.')

                json_img_path = sample['name'].replace('dataset/origin','dataset/auto_edit_json/img')
                sample['name'] = json_img_path
                sample['url'] = json_img_path
                img = cv2.imread(src_img_path)
                error_label_list = []
                for label in labels:
                    try: 
                        category = label['category']
                    except: 
                        category = 'error'

                    try: 
                        direction = label['attributes']['Direction']
                    except: 
                        direction = 'error'
                    try: 
                        type = label['attributes']['Type']
                    except: 
                        type = 'error'
                    try: 
                        continuity = label['attributes']['Continuity']
                    except: 
                        continuity = 'error'

                    vis_label = category + '_' +  direction + '_' +  type + '_' + continuity 
                    if not len(label['attributes']):
                        model = json.dumps(sample)
                        f1.write(model)
                        f1.write(',')
                        f1.write('\n')
                        error_label_list.append(label)
                        continue
                    
                    if label['category'] == 'road curb':
                        try:
                            label['attributes']['Direction']
                        except:
                            model = json.dumps(sample)
                            f1.write(model)
                            f1.write(',')
                            f1.write('\n')
                            error_label_list.append(label)
                            continue
                    else:
                        try:
                            label['attributes']['Continuity'] 
                            direct = label['attributes']['Direction']
                            type = label['attributes']['Type'] 
                        except:
                            model = json.dumps(sample)
                            f1.write(model)
                            f1.write(',')
                            f1.write('\n')  
                            error_label_list.append(label)
                            continue
                                        

                    if 'road curb' == label['category'] and 'NA' == direction:
                        model = json.dumps(sample)
                        f1.write(model)
                        f1.write(',')
                        f1.write('\n')
                        error_label_list.append(label)
                        continue
                    else:
                        if 'road curb' == label['category']:
                            continue
                        if 'NA' == category or 'NA' == direction or 'NA' == type or 'NA' == continuity or 'error' == category or 'error' == direction or 'error' == type or 'error' == continuity:
                            model = json.dumps(sample)
                            f1.write(model)
                            f1.write(',')
                            f1.write('\n')
                            error_label_list.append(label)
                            continue              
                    
                        if  'error' in vis_label.split('_'):
                            model = json.dumps(sample)
                            f1.write(model)
                            f1.write(',')
                            f1.write('\n')
                            error_label_list.append(label)

                if len(error_label_list) > 0:
                    for i in range(len(error_label_list)):
                        img = draw_err_ladmark(img,error_label_list[i],i)
                    cv2.imwrite(img_fix_name,img)
                
            f1.write(']')

                               

    #step 2: label visualization
    show_lane_voc_img(dstDir,lane_info)   #可视化
    #step 3: pascal voc -> yolo -------->>> todo
    convert_lane_pose_annotation_voc2yolo(dstDir,lane_info)

    #step 4: lane pose visualization
    vis_txt_lane_point(dstDir)

def seg_convert_main(srcDir,dstDir,srcjpgfolder):
    
    lane_classes = ('__background__', # always index 0
            'drivable', 'alternative', 'deceleration', 'crosswalk', 'No_driving_area')

    lane_direct = ['NA', 'Parallel', 'Vertical']

    lane_info = []
    lane_info.append(lane_classes)
    lane_info.append(lane_direct)


    # #step 1: scalabel lane annotation -> pascal xml
    i = 1
    for dirpath, dirnames, filenames in os.walk(srcDir):
        for filepath in filenames:
            fileName = os.path.join(dirpath,filepath)
            print(fileName)
            print("processing: {}, {}".format(i, fileName))
            i = i + 1
            xmlFileName = filepath[:-5] # remove ".json" 5 character
            # 解析该json文件，返回一个列表的列表，存储了一个json文件里面的所有方框坐标及其所属的类
            f = open(fileName)
            info = json.load(f)
            sam_list = info['frames']
            print('annotation sample num: ', len(sam_list))

            #创建修改json文件夹
            os.makedirs('./auto_edit_seg_json/json',exist_ok=True)
            fix_name = './auto_edit_seg_json/json/' + fileName.split('/')[-1]
            f1 = open(fix_name,'a')
            f1.write('[')

            save_fix_img_path = './auto_edit_seg_json/img'
            os.makedirs(save_fix_img_path,exist_ok=True)            

            for sample in sam_list:                                  
                objs = parseJson.parse_seg_Json(sample,lane_info) 
                xml_name = sample['name'].split('/')[-1]
                labels = sample['labels']

                # 如果存在检测对象，创建一个与该json文件具有相同名的VOC格式的xml文件
                if len(objs):
                    xml_voc_seg_writer(xml_name,dstDir,objs,1280,512, srcjpgfolder)                        
                else:
                    print(fileName)
                    continue
            
            for sample in sam_list:                                  
                objs = parseJson.parse_seg_Json(sample,lane_info) 
                xml_name = sample['name'].split('/')[-1]
                labels = sample['labels'] 
                src_img_path = sample['name'].replace('/items/dataset','.') 
                img_fix_name = sample['name'].replace('origin/','auto_edit_seg_json/img/') 
                img_fix_name = img_fix_name.replace('/items/dataset','.')

                json_img_path = sample['name'].replace('dataset/origin','dataset/auto_edit_seg_json/img')
                sample['name'] = json_img_path
                sample['url'] = json_img_path
                img = cv2.imread(src_img_path)
                error_label_list = []
                for label in labels:
                    try: 
                        category = label['category']
                    except: 
                        error_label_list.append(label)
                        category = 'error'

                    try: 
                        direction = label['attributes']['Direction']
                    except: 
                        error_label_list.append(label)
                        direction = 'error'

                    vis_label = category + '_' +  direction 
                    if not len(label['attributes']):
                        model = json.dumps(sample)
                        f1.write(model)
                        f1.write(',')
                        f1.write('\n')
                        error_label_list.append(label)
                        continue

                if len(error_label_list) > 0:
                    for i in range(len(error_label_list)):
                        img = draw_err_segmark(img,error_label_list[i],i)
                    cv2.imwrite(img_fix_name,img)
                
            f1.write(']')
                              
    # #step 2: label visualization
    is_src_show = True
    show_seg_voc_img(dstDir,lane_info,is_src_show)   #可视化
    
    # #step 3: pascal voc -> yolo --------目前训练不区别水平或垂直
    convert_seg_annotation_voc2yolo(dstDir,lane_info)

    # #step 4: lane pose visualization
    vis_txt_seg_point(dstDir)

def main(args):
    task = args.task

    srcDir = args.srcDir

    if 'object_detect' == task:
        object_convert_main(srcDir,args.task)
    if 'lane_detect' == task:
        lane_convert_main(srcDir)
    if 'segmentation' == task:
        seg_convert_main(srcDir)

def parse_args():
    parser = argparse.ArgumentParser(description='bdd100k dataset convert')
    parser.add_argument('--task', type=str,default='object_detect', help='object_detect ,lane_detect or segmentation')
    parser.add_argument('--srcDir', type=str,default='/home/robot/open_dataset/lane_dataset/BDD100K/bdd100k',
                        help='bdd annotation`s json file')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()    
    
    main(args)
