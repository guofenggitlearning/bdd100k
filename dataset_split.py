#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-21 15:18:07
lastTime     : 2024-10-21 17:45:47
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/dataset_split.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-05-31 18:44:23
lastTime     : 2024-08-23 16:54:24
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /data_process_project/dataset_split.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

import os
import random
import shutil

def dataset_split(dataset_root):
    #创建数据集保存路径
    yolo_save_data = dataset_root + '/yolo_format_data'
    os.makedirs(yolo_save_data+'/images/train',exist_ok=True)
    os.makedirs(yolo_save_data+'/images/val',exist_ok=True)
    os.makedirs(yolo_save_data+'/labels/train',exist_ok=True)
    os.makedirs(yolo_save_data+'/labels/val',exist_ok=True)

    img_folder = os.path.join(dataset_root,'images')
    img_list = os.listdir(img_folder)
    random.shuffle(img_list)

    index = 0
    for img_name in img_list:
        img_path = os.path.join(img_folder,img_name)
        txt_path = os.path.join(img_folder.replace('images','txt_Annotations'),img_name.replace('.jpg','.txt'))
        
        if index / len(img_list) < 0.90:
            shutil.copy(img_path,yolo_save_data+'/images/train/' + img_name)
            shutil.copy(txt_path,yolo_save_data+'/labels/train/' + img_name.replace('.jpg','.txt'))
        else:
            shutil.copy(img_path,yolo_save_data+'/images/val/' + img_name)
            shutil.copy(txt_path,yolo_save_data+'/labels/val/' + img_name.replace('.jpg','.txt'))

        index +=1
