#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-17 16:13:27
lastTime     : 2024-10-21 17:40:36
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/label2det_v1.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

import argparse
import json
import os
from os import path as osp
import sys


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir',type=str, default='/home/robot/open_dataset/lane_dataset/BDD100K/bdd100k/labels/100k/val', help='path to the label dir')
    parser.add_argument('--det_path',type=str, default='save_json/bdd100k_labels_images_val.json', help='path to output detection file')
    args = parser.parse_args()

    return args


def label2det(label, image_id):
    boxes = list()
    for frame in label['frames']:
        for obj in frame['objects']:
            if 'box2d' not in obj:
                continue
            xy = obj['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            box = {'name': label['name'],
                   'timestamp': frame['timestamp'],
                   'category': obj['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1,
                   'image_id': image_id}
            boxes.append(box)
    return boxes


def change_dir(mark_dir, det_mark_path, dataType):
    if not osp.exists(mark_dir):
        print('Can not find', mark_dir)
        return
    os.makedirs(os.path.join(det_mark_path),exist_ok=True)
    for data_set in dataType:
        label_dir = os.path.join(mark_dir,data_set)
        det_path = os.path.join(det_mark_path, 'bdd100k_labels_images_{}.json'.format(data_set))
        print('Processing', label_dir)
        input_names = [n for n in os.listdir(label_dir)
                    if osp.splitext(n)[1] == '.json']
        boxes = []
        count = 0
        for name in input_names:
            count += 1
            in_path = osp.join(label_dir, name)
            out = label2det(json.load(open(in_path, 'r')), count)
            boxes.append(out) 
                
            if count % 1000 == 0:
                print('Finished', count)
        with open(det_path, 'w') as fp:
            json.dump(boxes, fp, indent=4, separators=(',', ': '))


def main():
    args = parse_args()
    os.makedirs(args.det_path.split('/')[0],exist_ok=True)
    dataType = ['train', 'val']
    change_dir(args.label_dir, args.det_path,dataType)
