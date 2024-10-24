#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-21 15:25:03
lastTime     : 2024-10-24 08:28:39
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

from parseJson import parseJson,parse_lane_Json,parse_seg_Json
from utils.utils import xml_voc_lane_writer,show_lane_voc_img,convert_lane_pose_annotation_voc2yolo,vis_txt_lane_point,show_seg_voc_img
from utils.utils import xml_voc_seg_writer,convert_seg_annotation_voc2yolo,vis_txt_seg_point

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

def bdd2coco(srcDir, dataType, classes, task):

    #step 1: 将json合并
    merge_json_path = os.path.join(srcDir,task,'save_json')
    os.makedirs(merge_json_path,exist_ok=True)
    label_dir = os.path.join(srcDir,'labels','100k')
    change_dir(label_dir, merge_json_path,dataType)

    #step 2: 将合并后的json转换成coco格式
    attr_dict = dict()
    attr_dict["categories"] = []
    for cls in classes:
        attr_dict["categories"].append({"supercategories": "none", "id": 1,"name": "{}".format(cls)})
    # attr_dict["categories"] = [
    #     {"supercategory": "none", "id": 1, "name": "person"},
    #     {"supercategory": "none", "id": 2, "name": "rider"},
    #     {"supercategory": "none", "id": 3, "name": "car"},
    #     {"supercategory": "none", "id": 4, "name": "bus"},
    #     {"supercategory": "none", "id": 5, "name": "truck"},
    #     {"supercategory": "none", "id": 6, "name": "bike"},
    #     {"supercategory": "none", "id": 7, "name": "motor"},
    #     {"supercategory": "none", "id": 8, "name": "traffic light"},
    #     {"supercategory": "none", "id": 9, "name": "traffic sign"},
    #     {"supercategory": "none", "id": 10, "name": "train"}
    # ]

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
        bdd2coco_detection(attr_id_dict, attr_dict,train_labels, out_fn)
    
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

def object_convert_main(srcDir,task,classes):
    dataTypes = ['train', 'val']
  
    # #step 1: bdd100k2coco 
    bdd2coco(srcDir, dataTypes, classes, task) 

    # # step 2: coco2voc
    voc_root = coco2voc(srcDir, dataTypes, task)      

    # #step 3: label visualization
    show_voc_img(voc_root)   #可视化

    #step 4: voc2yolo
    convert_annotation_voc2yolo(voc_root)

    #step 5: dataset split and yolo format
    dataset_split(voc_root)

def lane_convert_main(srcDir,task,classes,lane_attribute_ok):
    from show_labels import get_lanes
    lane_classes = tuple(classes)
    lane_attri = ['solid','dashed']

    lane_direct = ['parallel', 'vertical']

    lane_info = []
    lane_info.append(lane_classes)
    lane_info.append(lane_attri)
    lane_info.append(lane_direct)
    dataTypes = ['train', 'val']
    img_w = 1280
    img_h = 720

    # # #step 1: scalabel lane annotation -> pascal xml
    for data_type in dataTypes:
        labels_path = os.path.join(srcDir,'labels/100k/{}'.format(data_type))
        labels_list = os.listdir(labels_path)
        for label_path in labels_list:
            f = open(os.path.join(labels_path,label_path))
            info = json.load(f)
            sam_list = info['frames'][0]['objects']
            objects = get_lanes(sam_list)
            
            xml_name = label_path.replace('.json', '.xml')
            
            xml_voc_lane_writer(xml_name,srcDir,objects,img_w,img_h)                        
    
    # # #step 2: label visualization
    # show_lane_voc_img(srcDir,lane_info)   #可视化
    # # #step 3: pascal voc -> yolo -------->>> todo
    convert_lane_pose_annotation_voc2yolo(srcDir,lane_info,lane_attribute_ok)

    # #step 4: lane pose visualization
    vis_txt_lane_point(srcDir,classes)

    # #step 5: dataset_split
    voc_root = os.path.join(srcDir,'lane_detection')
    dataset_split(voc_root)

def seg_convert_main(srcDir, task, seg_classes,seg_attribute_ok):
    from show_labels import get_areas
    lane_classes = seg_classes

    lane_info = []
    lane_info.append(lane_classes)

    dataTypes = ['train', 'val']
    img_w = 1280
    img_h = 720

    # #step 1: scalabel lane annotation -> pascal xml
    for data_type in dataTypes:
        labels_path = os.path.join(srcDir,'labels/100k/{}'.format(data_type))
        labels_list = os.listdir(labels_path)
        for label_path in labels_list:
            f = open(os.path.join(labels_path,label_path))
            info = json.load(f)
            sam_list = info['frames'][0]['objects']
            objects = get_areas(sam_list)
            xml_name = label_path.replace('.json', '.xml')            
            xml_voc_seg_writer(xml_name,srcDir,data_type,objects,img_w,img_h) 

    
                              
    # #step 2: label visualization
    is_src_show = True
    # show_seg_voc_img(srcDir,lane_info,is_src_show)   #可视化
    
    # #step 3: pascal voc -> yolo --------目前训练不区别水平或垂直
    convert_seg_annotation_voc2yolo(srcDir,lane_info)

    # #step 4: lane pose visualization
    vis_txt_seg_point(srcDir, task, seg_classes)

    # step 5: dataset split
    voc_root = os.path.join(srcDir,'segmentation')
    dataset_split(voc_root)

def main(args):
    task = args.task

    srcDir = args.srcDir

    if 'object_detect' == task:
        object_convert_main(srcDir,args.task,args.od_classes)
    if 'lane_detect' == task:
        lane_convert_main(srcDir,args.task,args.lane_classes,args.lane_attribute_ok)
    if 'segmentation' == task:
        seg_convert_main(srcDir, args.task, args.seg_classes,args.seg_attribute_ok)

def parse_args():
    parser = argparse.ArgumentParser(description='bdd100k dataset convert')
    parser.add_argument('--task', type=str,default='segmentation', help='object_detect ,lane_detect or segmentation')
    parser.add_argument('--srcDir', type=str,default='/home/robot/open_dataset/lane_dataset/BDD100K/bdd100k',
                        help='bdd annotation`s json file')
    parser.add_argument("--seg_classes", default=["area/drivable", "area/alternative", "area/unknown"],nargs="+", type=str,help="classes names")
    parser.add_argument("--od_classes", default=["person","rider","car","bus","truck","bike","motor","traffic light","traffic sign","train"],nargs="+", type=str,help="classes names")
    # parser.add_argument("--lane_classes", default=["lane/road curb",
    #                         "lane/single white",
    #                         "lane/double yellow",
    #                         "lane/single yellow",
    #                         "lane/crosswalk",
    #                         "lane/double white",
    #                         "lane/single other",
    #                         "lane/double other"],nargs="+", type=str,help="classes names")
    
    parser.add_argument("--lane_classes", default=['double yellow_parallel_solid', 'road curb_parallel_solid', 'single yellow_parallel_solid', 'single white_parallel_dashed', 'crosswalk_vertical_dashed', 'single white_parallel_solid', 'double white_parallel_solid', 'single white_vertical_solid', 'double yellow_parallel_dashed', 'single yellow_parallel_dashed', 'double white_parallel_dashed', 'crosswalk_vertical_solid', 'single white_vertical_dashed', 'single other_parallel_solid', 'single yellow_vertical_solid', 'road curb_parallel_dashed', 'road curb_vertical_solid', 'double yellow_vertical_solid', 'single other_parallel_dashed', 'crosswalk_parallel_solid', 'double other_parallel_solid', 'road curb_vertical_dashed', 'double white_vertical_solid', 'crosswalk_parallel_dashed', 'single other_vertical_solid', 'double yellow_vertical_dashed', 'single yellow_vertical_dashed', 'double other_parallel_dashed', 'double white_vertical_dashed'],nargs="+", type=str,help="classes names of lane_attribute_ok of False")
    parser.add_argument('--lane_attribute_ok', type=bool,default=False, help='lane_detect attribute')
    parser.add_argument('--seg_attribute_ok', type=bool,default=False, help='area_detect attribute')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()    
    
    main(args)
