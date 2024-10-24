#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-17 16:13:27
lastTime     : 2024-10-22 14:11:35
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/bdd2coco.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument(
          "-l", "--label_dir",
          default="/home/robot/project/pgf/bdd100k/bdd100k/save_json",
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default="coco_format",
          help="path to save coco formatted label file",
    )
    return parser.parse_args()


def bdd2coco_detection(id_dict, attr_dict, labeled_images, fn):
    
    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        image = dict()
        image['file_name'] = i[0]['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = i[0]['image_id']
        for obj in i:            
            counter = counter + 1
            empty_image = False

            annotation = dict()
            annotation["iscrowd"] = 0
            annotation["image_id"] = image['id']
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = obj['bbox'][2]
            y2 = obj['bbox'][3]
            annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            annotation['category_id'] = id_dict[obj['category']]
            annotation['ignore'] = 0
            annotation['id'] = counter
            annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            annotations.append(annotation)

        if empty_image:
            continue
        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)

    with open(fn, "w") as file:
        file.write(json_string)


# if __name__ == '__main__':

#     args = parse_arguments()
#     os.makedirs(args.save_path,exist_ok=True)
#     attr_dict = dict()
#     attr_dict["categories"] = [
#         {"supercategory": "none", "id": 1, "name": "person"},
#         {"supercategory": "none", "id": 2, "name": "rider"},
#         {"supercategory": "none", "id": 3, "name": "car"},
#         {"supercategory": "none", "id": 4, "name": "bus"},
#         {"supercategory": "none", "id": 5, "name": "truck"},
#         {"supercategory": "none", "id": 6, "name": "bike"},
#         {"supercategory": "none", "id": 7, "name": "motor"},
#         {"supercategory": "none", "id": 8, "name": "traffic light"},
#         {"supercategory": "none", "id": 9, "name": "traffic sign"},
#         {"supercategory": "none", "id": 10, "name": "train"}
#     ]

#     attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

#     # create BDD training set detections in COCO format
#     print('Loading training set...')
#     with open(os.path.join(args.label_dir,
#                            'bdd100k_labels_images_train.json')) as f:
#         train_labels = json.load(f)
#     print('Converting training set...')

#     out_fn = os.path.join(args.save_path,
#                           'bdd100k_labels_images_det_coco_train.json')
#     bdd2coco_detection(attr_id_dict, train_labels, out_fn)

#     print('Loading validation set...')
#     # create BDD validation set detections in COCO format
#     with open(os.path.join(args.label_dir,
#                            'bdd100k_labels_images_val.json')) as f:
#         val_labels = json.load(f)
#     print('Converting validation set...')

#     out_fn = os.path.join(args.save_path,
#                           'bdd100k_labels_images_det_coco_val.json')
#     bdd2coco_detection(attr_id_dict, val_labels, out_fn)
