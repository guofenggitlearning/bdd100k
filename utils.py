#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-22 15:52:25
lastTime     : 2024-10-24 08:45:53
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/utils/utils.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''
import sys
import os
import cv2
import shutil
import numpy as np
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw,ImageFont


def xml_voc_lane_writer(xml_name, srcDataPath, objects, resize_w, resize_h):
    from pascal_voc_lane_writer import Writer
    # create pascal voc writer (image_path, width, height)
    
    # for obj in objects:
    #             obj_category = obj['category'].split('/')[1]
    #             obj_direct = obj['attributes']['direction']
    #             obj_style = obj['attributes']['style']
    #             obj_poly = obj['poly2d']
    #             vis_label = category + '_' +  direction + '_' +  type + '_' + continuity
    writer = Writer(xml_name, resize_w, resize_h)
    
    for obj in objects:
        obj_poly = obj['poly2d']

        #获取车道线最大外接矩形、车道线点属性
        pt_min_x = obj_poly[0][0]
        pt_min_y = obj_poly[0][1]
        pt_max_x = obj_poly[0][0]
        pt_max_y = obj_poly[0][1] 
        pt_vis_list = []
        kpts = []
        for pt in obj_poly:
            pt_vis_list.append(pt[2])
            if pt_min_x > pt[0]:
                pt_min_x = pt[0]
            if pt_min_y > pt[1]:
                pt_min_y = pt[1]

            if pt_max_x < pt[0]:
                pt_max_x = pt[0]
            if pt_max_y < pt[1]:
                pt_max_y = pt[1]

            kpts.append([pt[0],pt[1]])
        
        bbox = [pt_min_x-2,pt_min_y-2,pt_max_x+2,pt_max_y+2]
        # <name>yellow</name>
        # <category>Single</category>
        # # <continuity>Full</continuity>
        # <direction>Parallel</direction>
        obj_name = obj['category'].split('/')[1]
        obj_direct = obj['attributes']['direction']
        obj_style = obj['attributes']['style']
        
        writer.addObject(obj_name, obj_direct, obj_style, pt_vis_list, kpts)    
    
    # write to file
    save_file_path = os.path.join(srcDataPath,'lane_detection','xmlAnnotation')
    os.makedirs(save_file_path,exist_ok=True)
    writer.save(save_file_path + '/' + xml_name)

    #保存原图
    save_img_path = os.path.join(srcDataPath,'lane_detection','images')
    os.makedirs(save_img_path,exist_ok=True)
    save_img_name = os.path.join(save_img_path,xml_name.replace('xml', 'jpg'))
    print('---------------')
    try:
        img_name = srcDataPath + '/images/100k/val/' + xml_name.replace('xml', 'jpg')
        shutil.copy(img_name, save_img_name)
    except:
        img_name = srcDataPath + '/images/100k/train/' + xml_name.replace('xml', 'jpg')
        shutil.copy(img_name, save_img_name)
    
    #print("do")

def show_lane_voc_img(dataset_path,classes):

    dataset_path = os.path.join(dataset_path,'lane_detection')
    lane_classes = classes[0]
    lane_attri = classes[1]
    lane_direct = classes[2]
    
    file_path_img =  dataset_path + '/images'
    file_path_xml =  dataset_path + '/xmlAnnotation'
    save_file_path = dataset_path + '/vis_jpeg'
    
    pathDir = os.listdir(file_path_img)
    for idx in range(len(pathDir)):   
        filename = pathDir[idx]
        
        try:
            tree = ET.parse(os.path.join(file_path_xml, filename.replace('.jpg','.xml')))
        except:
            print(filename)
        
        objs = tree.findall('object')        
        num_objs = len(objs)
        # boxes = np.zeros((num_objs, 5), dtype=np.uint16)
        poly2d = np.zeros((num_objs, 16), dtype=np.uint16)

        image_name = os.path.splitext(filename)[0]
        # print(image_name)
        
        try:
            img = Image.open(os.path.join(file_path_img, image_name + '.jpg')) 
        except:
            print(image_name)
            continue

        for ix, obj in enumerate(objs):
            lane = obj.find('poly2d').find('lane').text            
            cla = obj.find('name').text + '_' + obj.find('category').text + '_' + obj.find('continuity').text 
            # if(cla == "Stairs"):
            #     print(filename)
            #     print(cla)
            # print(cla)            
            print('=================================================',img.size)
            draw = ImageDraw.Draw(img)

            lane_split = lane.split(']')

            lane_line = []
            for ix in range(len(lane_split)):
                try:
                    point_x = int(float(lane_split[ix].split(',')[-2].split('[')[-1]))
                    point_y = int(float(lane_split[ix].split(',')[-1]))
                except:
                    continue                
                lane_line.append((point_x,point_y))  
                if(ix>0):
                    draw.line(lane_line[ix - 1] + lane_line[ix], fill=128, width=5) #线的起点和终点，线宽              
                
            
            # draw.regular_polygon(lane_line, outline=(255, 0, 0))  
            font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
            # draw.text([lane_line[0].x, lane_line[0].y], cla, 140, font=font) 
            draw.text([lane_line[1][0], lane_line[1][1]], cla, 140, font=font)        
            

        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        # cv2.imwrite(os.path.join(save_file_path, image_name + '.png'),img)
        img.save(os.path.join(save_file_path, image_name + '.jpg'))

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

def lane_convert(size, lane):

    dw = 1./size[0]
    dh = 1./size[1]
    
    keep_lane_line = []
    for idx in range(14):
        if idx >= len(lane):
            keep_lane_line.append(0.0)
            keep_lane_line.append(0.0)
            keep_lane_line.append(0.0)
        else:
            keep_lane_line.append(lane[idx][0] * dw)
            keep_lane_line.append(lane[idx][1] * dh)
            keep_lane_line.append(float(lane[idx][2]))

    return keep_lane_line

def convert_lane_pose_annotation(image_id, dataset_path,lane_info,lane_attribute_ok,new_lane_classess):
    classes = lane_info[0]   
    
    lane_classes    = lane_info[0]

    print(image_id)
    # image_id = image_id.split('.png')[0]
    # image_id = ["%s_" % tmp for tmp in image_id.split('.')[:-1]]
    try:
        in_file = open(dataset_path + '/xmlAnnotation/%s.xml'%(image_id))
    except:
        print(dataset_path + '/xmlAnnotation/%s.xml'%(image_id), '不存在！！！！')
    
    outpput_path = os.path.join(dataset_path,'txt_Annotations')
    
    os.makedirs(outpput_path,exist_ok=True)    
    
    out_file = open(outpput_path + '/%s.txt'%(image_id),'w') #生成txt格式文件
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')  
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):        
        
        cls = obj.find('name').text + '_' + obj.find('category').text + '_' + obj.find('continuity').text 

        if cls not in new_lane_classess:
            new_lane_classess.append(cls)

        # 读取车道线
        lane = obj.find('poly2d').find('lane').text
        lane_split = lane.split(']')

        lane_line = []
        type_c = obj.find('direction').text
        type_list = type_c.split(',')
        for ix in range(len(lane_split)):
            try:
                point_x = int(float(lane_split[ix].split(',')[-2].split('[')[-1]))
                point_y = int(float(lane_split[ix].split(',')[-1]))
                if type_list[ix].split('\'')[1] == 'L':
                    point_v = 1
                if type_list[ix].split('\'')[1] == 'C':
                    point_v = 2
                
            except:
                continue                
            lane_line.append((point_x,point_y,point_v))

        if(len(lane_line)>14):
            print("******************* lane point num: ",len(lane_line))
            exit(0)

        box_xmin = 0
        box_ymin = 0
        box_xmax = 0
        box_ymax = 0
        for idx in range(len(lane_line)):
            
            if idx == 0:
                box_xmin = lane_line[idx][0]
                box_ymin = lane_line[idx][1]
                box_xmax = lane_line[idx][0]
                box_ymax = lane_line[idx][1] 

            box_xmin = min(lane_line[idx][0],box_xmin)
            box_ymin = min(lane_line[idx][1],box_ymin)
            box_xmax = max(lane_line[idx][0],box_xmax)
            box_ymax = max(lane_line[idx][1],box_ymax)    

        try:
            if not lane_attribute_ok:
                # cls_src = 'lane/{}'.format(obj.find('name').text)
                cls_id = classes.index(cls)   
        except:
            continue
        b = (box_xmin, box_xmax, box_ymin, box_ymax)
        bb = convert((w,h), b)
       
        if lane_attribute_ok:
            lane_direct = obj.find('category')
            lane_style = obj.find('continuity')
            if lane_direct == 'parallel':
                direct_id = 1
            if lane_direct == 'vertical':
                direct_id = 2
            if lane_style == 'solid':
                direct_id = 3
            if lane_style == 'dashed':
                direct_id = 4
    
            ll = lane_convert((w,h),lane_line)
            # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(lll) for lll in ll]) + " "+direct_id+"\n")
        else:
                
            ll = lane_convert((w,h),lane_line)
            # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(lll) for lll in ll])+"\n")
    

def convert_lane_pose_annotation_voc2yolo(dataset_path,lane_info, lane_attribute_ok):
    
    dataset_path = os.path.join(dataset_path,'lane_detection')

    file_list = open(os.path.join(dataset_path,'test.txt'),'w')
    image_path = dataset_path + '/images'
    index_img = os.listdir(image_path)
    for id_img in index_img:
        file_list.write('%s\n'%(id_img))
    file_list.close()
    
    image_ids_train = open(os.path.join(dataset_path,'test.txt')).read().strip().split()
    list_file_train = open('save_test.txt', 'w')   
    new_lane_classess = []  
    for image_id in image_ids_train:
        list_file_train.write('./dataset-yolo/%s\n'%(image_id))
        print(image_id)  
        convert_lane_pose_annotation(image_id.split('.jpg')[0], dataset_path, lane_info,lane_attribute_ok,new_lane_classess)
    print('cls_num:',len(new_lane_classess))
    print('category:',new_lane_classess)
    

def vis_txt_lane_point(dstDir,classes):
    txtpath = os.path.join(dstDir,'lane_detection','txt_Annotations')
    txtlist = os.listdir(txtpath)
    for txt_temp in txtlist:

        f = open(os.path.join(txtpath,txt_temp),'r')
        lines = f.readlines()
        imgpath = os.path.join(dstDir,'lane_detection','images',txt_temp.replace('.txt','.jpg'))
        img = cv2.imread(imgpath)
        try:
            h, w, c = img.shape
        except:
            print(imgpath)
            exit(0)
        
        colors = [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]]
                
        for line in lines:
            print(line)
            l = line.split(' ')
            print(len(l))
            cla = classes[int(l[0])]
            cx = float(l[1]) * w
            cy = float(l[2]) * h
            weight = float(l[3]) * w
            height = float(l[4]) * h
            xmin = cx - weight/2
            ymin = cy - height/2
            xmax = cx + weight/2
            ymax = cy + height/2
            print((xmin,ymin),(xmax,ymax))
            cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
            
            kpts = []        
            for i in range(14):
                x = float(l[5:][3*i]) * w
                y = float(l[5:][3*i+1]) * h
                s = int(float(l[5:][3*i+2]))
                print(x,y,s)
                if s == 1.0:
                    cv2.circle(img,(int(x),int(y)),1,colors[i],2)
                    cv2.putText(img,'L',(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX, 0.5, colors[i], 1)
                elif s == 2.0:
                    cv2.circle(img,(int(x),int(y)),1,colors[i],2)
                    cv2.putText(img,'C',(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX, 0.5, colors[i], 1)
                kpts.append([int(x),int(y),int(s)])
            print(kpts)

            #折线图
            # kpt_line = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                        # [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
            kpt_line = [[14,13],[13,12],[12,11],[11,10],[10,9],[9,8],[8,7],[7,6],[6,5],[5,4],[4,3],
                        [3,2],[2,1]]

            for j in range(len(kpt_line)):
                m,n = kpt_line[j][0],kpt_line[j][1]
                if kpts[m-1][2] !=0 and kpts[n-1][2] !=0:
                    cv2.line(img,(kpts[m-1][0],kpts[m-1][1]),(kpts[n-1][0],kpts[n-1][1]),colors[j],2)
            cv2.putText(img,cla,(int(xmin)+10,int(ymin)+10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        
        save_path = os.path.join(dstDir,'lane_detection','txt_visualization')
        os.makedirs(save_path,exist_ok=True)
        save_img_path = os.path.join(save_path,txt_temp.replace('.txt','.jpg'))
        cv2.imwrite(save_img_path,img)

def seg_convert(size, lane):
    dw = 1./size[0]
    dh = 1./size[1]
    
    keep_lane_line = []
    for idx in range(len(lane)):
        keep_lane_line.append(lane[idx][0] * dw)
        keep_lane_line.append(lane[idx][1] * dh)

    return keep_lane_line

def convert_seg_annotation(image_id, dataset_path,lane_info):
        
    lane_classes    = lane_info[0]

    print(image_id)
    image_id = image_id.split('.png')[0]
    # image_id = ["%s_" % tmp for tmp in image_id.split('.')[:-1]]
    try:
        in_file = open(dataset_path + '/segAnnotations/%s.xml'%(image_id))
    except:
        print(dataset_path + '/segAnnotations/%s.xml'%(image_id), '不存在！！！！')
    
    outpput_path = os.path.join(dataset_path,'txt_Annotations')
    
    os.makedirs(outpput_path,exist_ok=True)    
    
    out_file = open(outpput_path + '/%s.txt'%(image_id),'w') #生成txt格式文件
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')  
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        #路面分割区域不分水平和垂直处理。
        cls_name = obj.find('category').text        

        # 读取路面区域
        lane = obj.find('poly2d').find('mask').text
        lane_split = lane.split(']')

        type = obj.find('direction')
        type_c = obj.find('direction').text
        type_list = type_c.split(',')
        
        lane_area = []
        for ix in range(len(lane_split)):
            try:
                point_x = int(float(lane_split[ix].split(',')[-2].split('[')[-1]))
                point_y = int(float(lane_split[ix].split(',')[-1]))

                if type_list[ix].split('\'')[1] == 'L':
                    point_v = 1
                if type_list[ix].split('\'')[1] == 'C':
                    point_v = 2
            except:
                continue               
            lane_area.append((point_x,point_y,point_v))

        box_xmin = 0
        box_ymin = 0
        box_xmax = 0
        box_ymax = 0
        for idx in range(len(lane_area)):
            point_x = lane_area[idx][0]
            point_y = lane_area[idx][1]

            if idx == 0:
                box_xmin = point_x
                box_ymin = point_y
                box_xmax = point_x
                box_ymax = point_y

            box_xmin = min(point_x,box_xmin)
            box_ymin = min(point_y,box_ymin)
            box_xmax = max(point_x,box_xmax)
            box_ymax = max(point_y,box_ymax) 

        if(len(lane_area)>30):
            print("******************* lane point num: ",len(lane_area))
            exit(0)
        
        try:
            cls_id = lane_classes.index(cls_name) #减去背景   
        except:
            continue
        b = (int(box_xmin), int(box_xmax), int(box_ymin), int(box_ymax))
        bb = convert((w,h), b)
        ll = seg_convert((w,h),lane_area)

        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(lll) for lll in ll]) + "\n")

def convert_seg_annotation_voc2yolo(dataset_path,lane_info):

    dataset_path = os.path.join(dataset_path,'segmentation')
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
        convert_seg_annotation(image_id.split('.jpg')[0], dataset_path, lane_info)

def show_seg_voc_img(dataset_path,classes,is_src_show=True):
    lane_classes = classes[0]
    
    dataset_path = os.path.join(dataset_path,'segmentation')
    file_path_img =  dataset_path + '/images'
    file_path_xml =  dataset_path + '/segAnnotations'
    save_file_path = dataset_path + '/vis_seg_jpeg'
    
    pathDir = os.listdir(file_path_img)
    for idx in range(len(pathDir)):   
        filename = pathDir[idx]
        
        try:
            tree = ET.parse(str(os.path.join(file_path_xml, filename.replace('.jpg','.xml'))))
        except:
            print(filename)
        root = tree.getroot()
        size = root.find('size')
        if size:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
        else:
            width, height, depth = -1, -1, -1

        objs = root.findall('object')        
        num_objs = len(objs)
        image_name = os.path.splitext(filename)[0]
        if not is_src_show:
            img = Image.new('RGB', (width, height), (0, 0, 0))

        else:            
            # print(image_name)        
            try:
                img = Image.open(os.path.join(file_path_img, image_name + '.jpg')) 
            except:
                print(image_name)
                continue

        for ix, obj in enumerate(objs):
            mask = obj.find('poly2d').find('mask').text            
            cla = obj.find('category').text           
            print('=================================================',img.size)
            draw = ImageDraw.Draw(img)

            mask_split = mask.split(']')

            mask_line = []
            for ix in range(len(mask_split)):
                try:
                    point_x = int(float(mask_split[ix].split(',')[-2].split('[')[-1]))
                    point_y = int(float(mask_split[ix].split(',')[-1]))
                except:
                    continue                
                mask_line.append((point_x,point_y))  
                if(ix>0):
                    draw.line(mask_line[ix - 1] + mask_line[ix], fill=128, width=5) #线的起点和终点，线宽

            draw.line(mask_line[0] + mask_line[-1],fill=5)            
                
            
            # draw.regular_polygon(lane_line, outline=(255, 0, 0))  
            font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
            # draw.text([lane_line[0].x, lane_line[0].y], cla, 140, font=font) 
            draw.text([mask_line[1][0], mask_line[1][1]], cla, 140, font=font)        
            

        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        # cv2.imwrite(os.path.join(save_file_path, image_name + '.png'),img)
        img.save(os.path.join(save_file_path, image_name + '.jpg'))

def vis_txt_seg_point(srcDir, task, seg_classess):

    dataset_path = os.path.join(srcDir,'segmentation')
    txtpath = os.path.join(dataset_path,'txt_Annotations')
    txtlist = os.listdir(txtpath)
    for txt_temp in txtlist:

        f = open(os.path.join(txtpath,txt_temp),'r')
        lines = f.readlines()
        imgpath = os.path.join(dataset_path,'images',txt_temp.replace('.txt','.jpg'))
        img = cv2.imread(imgpath)
        h, w, c = img.shape
        
        colors = [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]]
                
        for line in lines:
            print(line)
            l = line.split(' ')
            print(len(l))
            cla = seg_classess[int(l[0])]
            cx = float(l[1]) * w
            cy = float(l[2]) * h
            weight = float(l[3]) * w
            height = float(l[4]) * h
            xmin = cx - weight/2
            ymin = cy - height/2
            xmax = cx + weight/2
            ymax = cy + height/2
            print((xmin,ymin),(xmax,ymax))
            cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
            
            kpts = []        
            for i in range(int((len(l) - 5)/2) ):
                x = int(float(l[5 + 2*i+0]) * w)
                y = int(float(l[5 + 2*i+1]) * h)
                print(x,y)
                cv2.circle(img,(int(x),int(y)),1,colors[i],2)
                kpts.append([int(x),int(y)])
            print(kpts)            

            for i in range(len(kpts)):
                x = kpts[i][0]
                y = kpts[i][1]
                if i >0:
                   cv2.line(img,(kpts[i][0],kpts[i][1]),(kpts[i-1][0],kpts[i-1][1]),colors[i],2) 

            cv2.line(img,(kpts[0][0],kpts[0][1]),(kpts[-1][0],kpts[-1][1]),(255,0,0),2)
            cv2.putText(img,cla,(int(xmin)+10,int(ymin)+10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)

        img = cv2.resize(img, None, fx=0.9, fy=0.9)
        save_path = os.path.join(srcDir,'txt_visualization')
        os.makedirs(save_path,exist_ok=True)
        save_img_path = os.path.join(save_path,txt_temp.replace('.txt','.jpg'))
        cv2.imwrite(save_img_path,img)

def xml_voc_seg_writer(xml_name, srcDir, task, objects, resize_w, resize_h):
    from pascal_voc_seg_writer import Writer
    # create pascal voc writer (image_path, width, height)
    data_root = os.path.join(srcDir,'segmentation')
    writer = Writer(xml_name.replace(".xml",".jpg"), resize_w, resize_h)
    
    for obj in objects:
        obj_poly = obj['poly2d']
        cls = obj['category']
        #获取车道线最大外接矩形、车道线点属性
        pt_min_x = obj_poly[0][0]
        pt_min_y = obj_poly[0][1]
        pt_max_x = obj_poly[0][0]
        pt_max_y = obj_poly[0][1] 
        pt_vis_list = []
        kpts = []
        for pt in obj_poly:
            pt_vis_list.append(pt[2])
            if pt_min_x > pt[0]:
                pt_min_x = pt[0]
            if pt_min_y > pt[1]:
                pt_min_y = pt[1]

            if pt_max_x < pt[0]:
                pt_max_x = pt[0]
            if pt_max_y < pt[1]:
                pt_max_y = pt[1]

            kpts.append([pt[0],pt[1]])
        
        bbox = [pt_min_x,pt_min_y,pt_max_x,pt_max_y]
        
        obj_name = obj['category']
        
        writer.addObject(obj_name, pt_vis_list, kpts) 
    
    # write to file
    os.makedirs(data_root+'/segAnnotations',exist_ok=True)
    writer.save(data_root +"/segAnnotations"+'/'+ xml_name.replace('.jpg','.xml'))

    #保存原图
    img_name = srcDir + '/images/100k/{}/'.format(task) + xml_name.replace('xml','jpg')
    os.makedirs(os.path.join(data_root,'images'),exist_ok=True)
    save_img_name = os.path.join(data_root,'images') + '/' + xml_name.replace('xml','jpg')
    print('---------------')
    print(img_name)
    print(save_img_name)
    shutil.copy(img_name, save_img_name)
