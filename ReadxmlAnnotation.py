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
            new_xmax = w-int(box.find("bndbox/xmin").text)
            #new_ymin = h-int(box.find("bndbox/ymin").text)
            new_xmin = w-int(box.find("bndbox/xmax").text)
            #new_ymax = h-int(box.find("bndbox/ymax").text)
            box.find("bndbox/xmin").text = str(new_xmin)
            box.find("bndbox/xmax").text = str(new_xmax)
            #box.find("bndbox/ymin").text = str(new_ymin)
            #box.find("bndbox/ymax").text = str(new_ymax)
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
    h_new, w_new = max(int(h*rescale_ratio),1), max(int(w*rescale_ratio),1)
    try:
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    except:
        print("WHAT???")
    return img
def img_add(src_img, main_img, box, tree):
    s_h, s_w, _, = src_img.shape
    m_h, m_w, _, = main_img.shape
    h = s_h*2 + m_h
    w = s_w*2 + m_w
    canvas = np.zeros((h,w,3), dtype = np.uint8)
    canvas[s_h:s_h+m_h, s_w:s_w+m_w] = main_img
    
    min_y = int(s_h)
    max_y = max(int(m_h), min_y+1)
    min_x = int(s_w)
    max_x = max(int(m_w), min_x+1)
    new_x = np.random.randint(low=min_x, high=max_x)
    new_y = np.random.randint(low=min_y, high=max_y)

    canvas[new_y:new_y+s_h, new_x:new_x+s_w] = src_img
    output_img = canvas[s_h:s_h+m_h, s_w:s_w+m_w]
    #cv2.imwrite()


    local_xmin = max(0, new_x-s_w)
    local_xmax = min(m_w, new_x)
    local_ymin = max(0, new_y-s_h)
    local_ymax = min(m_h, new_y)
    #cv2.rectangle(output_img,(local_xmin,local_ymin),(local_xmax,local_ymax),(255,0,0),2)
   
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

     # remove overlap
    box_count = 0
    for box in root.iter('object'):
        box_count+=1
    actual_count = 0
    boxes = [i for i in root.iter('object')]
    for box in boxes:
        actual_count += 1
        xmin = int(box.find("bndbox/xmin").text)
        ymin = int(box.find("bndbox/ymin").text)
        xmax = int(box.find("bndbox/xmax").text)
        ymax = int(box.find("bndbox/ymax").text)
        dx = max(min(xmax, local_xmax) - max(xmin, local_xmin), 0)
        dy = max(min(ymax, local_ymax) - max(ymin, local_ymin), 0)
        #cv2.rectangle(output_img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
        #cv2.imshow("Show",output_img)
        #cv2.waitKey()  
        overlap = dx*dy
        box_area = (xmax-xmin)*(ymax-ymin)
        #ET.SubElement(box, "AREA").text = str(overlap)+'==='+str(box_area)
        #ET.dump(root)
        #print(overlap, box_area)
        #print("HEY")
        if overlap >= box_area*0.6:
            #
            
            root.remove(box)
            #ET.dump(root)
            #print("HEY")
            #ET.SubElement(box, 'OVER')

    if actual_count!=box_count:
        print(actual_count, box_count)
    root.append(new_box)
    output_annotation_path
    #ET.dump(new_box)
    
    ET.indent(root)
    #ET.dump(root)
    return output_img
    

count = 10000
def copy_paste(box, target_main):
    # read img
    src_img = crop_img(box)
    main_img = cv2.imread(os.path.join(train_Image_path, target_main+'.jpg'))
    # random flip
    src_img, main_img, tree = random_flip_horizontal(src_img, main_img, target_main)
    # Large Scale Jittering
    src_img = Large_Scale_Jittering(src_img)
    # for the last
    #cv2.imshow("MYI",src_img)
    #cv2.waitKey(0)
    img = img_add(src_img, main_img, box, tree)
    global count 
    count += 1
    xml_path = os.path.join(output_annotation_path, '{:0>6d}'.format(count)+'.xml')
    jpg_path = os.path.join(output_Image_path, '{:0>6d}'.format(count)+'.jpg')

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
    target_list = [i.split('.')[0] for i in os.listdir(train_annotation_path)]
    global box_list
    
    # read all box in train dataset
    for target in tqdm(target_list):
        target_xml_path = os.path.join(train_annotation_path, target + '.xml')
        #target_Image_path = os.path.join(train_Image_path, target + '.jpg')
        content = read_content(target_xml_path)
        box_list += content
        #box_dict[target] = content
    #print(box_list)
    #print(box_dict)
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output\JPEGImages'):
        os.mkdir('output\JPEGImages')
    if not os.path.exists('output\Annotations'):
        os.mkdir('output\Annotations')
    # for all box, randomlypaste onto a pic
    for box in tqdm(box_list):
        target_main = np.random.choice(target_list)
        copy_paste(box, target_main)



if __name__=='__main__':
    main()
    