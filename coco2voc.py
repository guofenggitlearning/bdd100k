#!/usr/bin/env python
# coding=utf-8
'''
creater      : PGF
since        : 2024-10-21 13:53:47
lastTime     : 2024-10-21 19:18:01
LastAuthor   : PGF
message      : The function of this file is 
文件相对于项目的路径   : /bdd100k/coco2voc.py
Copyright (c) 2024 by pgf email: nchu_pgf@163.com, All Rights Reserved.
'''

'''
把coco数据集合的所有标注转换到voc格式，不改变图片命名方式，
注意，原来有一些图片是黑白照片，检测出不是 RGB 图像，这样的图像不会被放到新的文件夹中
'''
from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image

# 生成图片保存的路径
CKimg_dir = './coco2024_voc/images'
# 生成标注文件保存的路径
CKanno_dir = './coco2024_voc/annotations'


# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path,exist_ok=True)
    else:
        os.makedirs(path,exist_ok=True)


def save_annotations(filename, objs, save_path, filepath):
    annopath = save_path[1] + "/" + filename + ".xml"  # 生成的xml文件保存路径
    dst_path = save_path[0] + "/" + filename + ".jpg"
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, dataType, img, classes, origin_image_dir, save_path,verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, '100k', dataType, filename + '.jpg')
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs, save_path, filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)


def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5(origin_anno_dir, origin_image_dir, save_path, verbose=False):
    dataTypes = ['train', 'val']
    for dataType in dataTypes:
        annFile = 'bdd100k_labels_images_det_coco_{}.json'.format(dataType)
        annpath = os.path.join(origin_anno_dir, annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(coco, dataType, img, classes, origin_image_dir, save_path, verbose)


def main():
    base_dir = './coco2024_voc'  # step1 这里是一个新的文件夹，存放转换后的图片和标注
    image_dir = os.path.join(base_dir, 'images')  # 在上述文件夹中生成images，annotations两个子文件夹
    anno_dir = os.path.join(base_dir, 'annotations')
    mkr(image_dir)
    mkr(anno_dir)
    origin_image_dir = '/home/robot/open_dataset/lane_dataset/BDD100K/bdd100k/images'  # step 2原始的coco的图像存放位置
    origin_anno_dir = 'coco_format'  # step 3 原始的coco的标注存放位置
    print(origin_anno_dir)
    verbose = False  # 是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(origin_anno_dir, origin_image_dir, verbose)


if __name__ == "__main__":
    main()
