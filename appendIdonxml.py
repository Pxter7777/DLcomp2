from math import trunc
from multiprocessing.sharedctypes import Value
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import cv2
from tqdm import tqdm, trange
import numpy as np

NametoId = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}
origin_xml_path = os.path.join('.', 'output', "Annotations")
output_xml_path = os.path.join('.', 'output', "Annotations_with_ID")
#origin_xml_path = os.path.join('.', 'datalab-2021-cup2-object-detection', 'VOCdevkit_train', 'VOC2007', 'Annotations')
#output_xml_path = os.path.join('.', 'datalab-2021-cup2-object-detection', 'VOCdevkit_train', 'VOC2007', 'Annotations_with_ID')
if not os.path.exists(output_xml_path):
    os.mkdir(output_xml_path)
def read_content(target: str):
    xml_path = os.path.join(origin_xml_path, target+'.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    list_with_all_boxes = []
    
    for box in root.iter('object'):
        name = box.find('name').text
        Id = ET.Element('ID')
        Id.text = str(NametoId.get(name, -1))
        box.insert(0, Id)
    new_xml_path = os.path.join(output_xml_path, target+'.xml')
    ET.indent(root)
    tree.write(new_xml_path)

target_list = [i.split('.')[0] for i in os.listdir(origin_xml_path )]
global box_list

# read all box in train dataset
for target in tqdm(target_list):
    #target_xml_path = os.path.join(train_annotation_path, target + '.xml')
    #target_Image_path = os.path.join(train_Image_path, target + '.jpg')
    read_content(target)
