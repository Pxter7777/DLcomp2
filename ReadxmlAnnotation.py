from math import trunc
import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm, trange
def read_content(xml_file: str, jpg_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    
    for boxes in root.iter('object'):
        filename = root.find('filename').text
        #ymin, xmin, ymax, xmax = None, None, None, None
        name = boxes.find('name').text
        pose = boxes.find('pose').text
        truncated = int(boxes.find('truncated').text)
        difficult = int(boxes.find('difficult').text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        try:
            src_img = crop_img(xmin, ymin, xmax, ymax, jpg_file)
        except:
            src_img = None
        box = {
            'name':name,
            'pose':pose,
            'truncated':truncated,
            'difficult':difficult,
            'xmin':xmin,
            'ymin':ymin,
            'xmax':xmax,
            'ymax':ymax,
            'src_img':src_img
        }
        list_with_all_boxes.append(box)

    return filename, list_with_all_boxes
def crop_img(xmin, ymin, xmax, ymax, jpg_file):
    img = cv2.imread(jpg_file)
    return img[ymin:ymax, xmin:xmax]


def main():
    data_folder_path = os.path.join('.', 'datalab-2021-cup2-object-detection')
    train_annotation_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'Annotations')
    train_Image_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'JPEGImages')
    
    target_list = [os.path.splitext(i)[0] for i in os.listdir(train_annotation_path)]

    for target in tqdm(target_list):
        target_xml_path = os.path.join(train_annotation_path, target + '.xml')
        target_Image_path = os.path.join(train_Image_path, target + '.jpg')
        name, boxes = read_content(target_xml_path, target_Image_path)
    #print(name, boxes)




if __name__=='__main__':
    main()
    