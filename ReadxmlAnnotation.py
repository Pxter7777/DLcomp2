from math import trunc
import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm, trange
import numpy as np

def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    
    for boxes in root.iter('object'):
        box = {
            'target':root.find('filename').text.split('.')[0],
            'name':boxes.find('name').text,
            'pose':boxes.find('pose').text,
            'truncated':int(boxes.find('truncated').text),
            'difficult':int(boxes.find('difficult').text),
            'xmin':int(boxes.find("bndbox/xmin").text),
            'ymin':int(boxes.find("bndbox/ymin").text),
            'xmax':int(boxes.find("bndbox/xmax").text),
            'ymax':int(boxes.find("bndbox/ymax").text)
        }
        list_with_all_boxes.append(box)
    return list_with_all_boxes
def crop_img(box):
    jpg_path = os.path.join(train_Image_path, box['target']+'.jpg')
    img = cv2.imread(jpg_path)
    return img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
def copy_paste(box, target_main):
    src_img = crop_img(box)
    main_img = cv2.imread(os.path.join(train_Image_path, target_main+'.jpg'))


data_folder_path = os.path.join('.', 'datalab-2021-cup2-object-detection')
train_annotation_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'Annotations')
train_Image_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'JPEGImages')

def main():
    #target_list = [os.path.splitext(i)[0] for i in os.listdir(train_annotation_path)]
    target_list = [i.split('.')[0] for i in os.listdir(train_annotation_path)]
    box_list = []
    # read all box in train dataset
    for target in tqdm(target_list):
        target_xml_path = os.path.join(train_annotation_path, target + '.xml')
        #target_Image_path = os.path.join(train_Image_path, target + '.jpg')
        box_list += read_content(target_xml_path)
    #print(box_list)
    if not os.path.exists('output'):
        os.mkdir('output')
    # for all box, randomlypaste onto a pic
    for box in tqdm(box_list[:100]):
        target_main = np.random.choice(target_list)
        copy_paste(box, target_main)



if __name__=='__main__':
    main()
    