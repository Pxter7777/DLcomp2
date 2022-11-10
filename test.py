from math import trunc
from multiprocessing.sharedctypes import Value
import xml.etree.ElementTree as ET
import xml.dom.minidom
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
            'truncated':boxes.find('truncated').text,
            'difficult':boxes.find('difficult').text,
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

def xml_tree(target):
    target_xml_path = os.path.join(train_annotation_path, target + '.xml')
    tree = ET.parse(target_xml_path)
    #root = tree.getroot()
    return tree

def random_flip_horizontal(src_img, main_img, target_main):
    if np.random.random() < 0.8:
        src_img = src_img[:,::-1,:]
    tree = xml_tree(target_main)
    root = tree.getroot()
    if np.random.random() < 0.5:
        main_img = main_img[:,::-1,:]
        w = int(root.find('size/width').text)
        h = int(root.find('size/height').text)
        for box in root.iter('object'):
            new_xmin = w-int(box.find("bndbox/xmin").text)
            new_ymin = h-int(box.find("bndbox/ymin").text)
            new_xmax = w-int(box.find("bndbox/xmax").text)
            new_ymax = h-int(box.find("bndbox/ymax").text)
            box.find("bndbox/xmin").text = str(new_xmin)
            box.find("bndbox/xmax").text = str(new_xmax)
            box.find("bndbox/ymin").text = str(new_ymin)
            box.find("bndbox/ymax").text = str(new_ymax)
            #box.set('bndbox/xmin', str(new_xmin))
            #box.set('bndbox/xmax', str(new_xmax))
            #box.set('bndbox/ymin', str(new_ymin))
            #box.set('bndbox/ymax', str(new_ymax))
    #ET.dump(root)
    return src_img, main_img, tree
def Large_Scale_Jittering(img):
    rescale_ratio = np.random.uniform(0.1, 2.0)
    h, w, _, = img.shape
    # rescale
    h_new, w_new = int(h*rescale_ratio), int(w*rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return img
def img_add(src_img, main_img, box, tree):
    s_h, s_w, _, = src_img.shape
    m_h, m_w, _, = main_img.shape
    h = s_h*2 + m_h
    w = s_w*2 + m_w
    canvas = np.zeros((h,w,3), dtype = np.uint8)
    canvas[s_h:s_h+m_h, s_w:s_w+m_w] = main_img
    
    min_y = int(s_h)
    max_y = int(m_h)
    min_x = int(s_w)
    max_x = int(m_w)
    new_x = np.random.randint(low=min_x, high=max_x)
    new_y = np.random.randint(low=min_y, high=max_y)

    canvas[new_y:new_y+s_h, new_x:new_x+s_w] = src_img
    output_img = canvas[s_h:s_h+m_h, s_w:s_w+m_w]
    #cv2.imwrite()


    local_xmin = max(0, new_x-s_w)
    local_xmax = min(m_w, new_x)
    local_ymin = max(0, new_y-s_h)
    local_ymax = min(m_w, new_y)
    
    new_box = ET.Element('object')
    #ET.dump(new_box)
    
    ET.SubElement(new_box, 'name').text = box['name']
    ET.SubElement(new_box, 'pose').text = box['pose']
    ET.SubElement(new_box, 'truncated').text = box['truncated']
    ET.SubElement(new_box, 'difficult').text = box['difficult']
    bndbox = ET.SubElement(new_box, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(local_xmin)
    ET.SubElement(bndbox, 'ymin').text = str(local_ymin)
    ET.SubElement(bndbox, 'xmax').text = str(local_xmax)
    ET.SubElement(bndbox, 'ymax').text = str(local_ymax)

    root = tree.getroot()
    root.append(new_box)
    output_annotation_path
    #ET.dump(new_box)
    
    ET.indent(root)
    #ET.dump(root)
    return output_img
    

count = 0 
def copy_paste(box, target_main):
    # read img
    src_img = crop_img(box)
    main_img = cv2.imread(os.path.join(train_Image_path, target_main+'.jpg'))
    # random flip
    src_img, main_img, tree = random_flip_horizontal(src_img, main_img, target_main)
    # Large Scale Jittering
    src_img = Large_Scale_Jittering(src_img)
    # for the last
    cv2.imshow("MYI",src_img)
    cv2.waitKey(0)
    img = img_add(src_img, main_img, box, tree)
    global count 
    count += 1
    xml_path = os.path.join(output_annotation_path, str(count)+'.xml')
    jpg_path = os.path.join(output_Image_path, str(count)+'.jpg')

    tree.write(xml_path)
    cv2.imwrite(jpg_path, img)
    # flip main
    # randomly place pic
        # pick place
        # generate pic and 


data_folder_path = os.path.join('.', 'datalab-2021-cup2-object-detection')
train_annotation_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'Annotations')
train_Image_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'JPEGImages')
output_annotation_path = os.path.join('.', 'output', 'Annotations')
output_Image_path = os.path.join('.', 'output', 'JPEGImages')
box_list = []
box_dict = {}

def main():
    #target_list = [os.path.splitext(i)[0] for i in os.listdir(train_annotation_path)]
    target_list = [i.split('.')[0] for i in os.listdir(output_annotation_path)]
    global box_list

    # read all box in train dataset
    for target in tqdm(target_list):
        print(target)
        target_xml_path = os.path.join(output_annotation_path, target + '.xml')
        #target_Image_path = os.path.join(train_Image_path, target + '.jpg')
        content = read_content(target_xml_path)
        jpg_path = os.path.join(output_Image_path, target + '.jpg')
        img = cv2.imread(jpg_path)
        #cv2.imshow(img)
        for b in content:
            cv2.rectangle(img,(b['xmin'],b['ymin']),(b['xmax'],b['ymax']),(255,255,255),5)
        cv2.imshow("Show",img)
        cv2.waitKey()  
        #box_list += content





if __name__=='__main__':
    main()
    