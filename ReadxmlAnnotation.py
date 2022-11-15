from math import trunc
from multiprocessing.sharedctypes import Value
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import cv2
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

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

def read_content(target: str):
    target_xml_path = os.path.join(train_annotation_path, target + '.xml')
    tree = ET.parse(target_xml_path)
    root = tree.getroot()
    list_with_all_boxes = []
    global box_pool, Record, img_pool
    box_count_list = [0]*20
    for boxes in root.iter('object'):
        name = boxes.find('name').text
        Id = NametoId.get(name, -1)
        #Record[Id] += 1
        box = {
            'target':root.find('filename').text.split('.')[0],
            'name':name,
            'Id': Id,
            'pose':boxes.find('pose').text,
            'truncated':boxes.find('truncated').text,
            'difficult':boxes.find('difficult').text,
            'xmin':int(boxes.find("bndbox/xmin").text),
            'ymin':int(boxes.find("bndbox/ymin").text),
            'xmax':int(boxes.find("bndbox/xmax").text),
            'ymax':int(boxes.find("bndbox/ymax").text)
        }
        list_with_all_boxes.append(box)
        box_pool = box_pool.append(box, ignore_index=True)
        Record[Id] += 1
        box_count_list[Id] += 1
    img_pool.loc[target] = box_count_list
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

def img_clone(src_img, main_img, box, tree):
    
    global Record

    s_h, s_w, _, = src_img.shape
    m_h, m_w, _, = main_img.shape
    h = s_h*2 + m_h
    w = s_w*2 + m_w
    canvas = np.zeros((h,w,3), dtype = np.uint8)
    canvas[s_h:s_h+m_h, s_w:s_w+m_w] = main_img

    #canvas = im
    mask = 255*np.ones(src_img.shape, src_img.dtype)
    if (mask.shape[0]<=4) or (mask.shape[1]<=4):
        return None
    h_var = np.random.randint(low=int(s_h*1.5), high=max(int(s_h*0.5+m_h),int(s_h*1.5)+1))
    w_var = np.random.randint(low=int(s_w*1.5), high=max(int(s_w*0.5+m_w),int(s_w*1.5)+1))

    center = (w_var, h_var)
    try:
        normal_clone = cv2.seamlessClone(src_img, canvas, mask, center, cv2.NORMAL_CLONE)
    except:
        print("WTF")
    output_img = normal_clone[s_h:s_h+m_h, s_w:s_w+m_w]
    #cv2.imshow('o', src_img)
    #cv2.waitKey()
    #cv2.imshow('o', mask)
    #cv2.waitKey()
    #cv2.imshow('o', canvas)
    #cv2.waitKey()
    #cv2.imshow('o', output_img)
    #cv2.waitKey()
    # modify annotation
    local_xmin = max(0, int(w_var-1.5*s_w))
    local_xmax = min(m_w, int(w_var-0.5*s_w))
    local_ymin = max(0, int(h_var-1.5*s_h))
    local_ymax = min(m_h, int(h_var-0.5*s_h))
   
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

        overlap = dx*dy
        box_area = (xmax-xmin)*(ymax-ymin)
        if overlap >= box_area*0.6:
            root.remove(box)
        else:
            Id = NametoId.get(box.find("name").text, -1)
            Record[Id] += 1


    if actual_count!=box_count:
        print(actual_count, box_count)
    root.append(new_box)
    
    ET.indent(root)
    #ET.dump(root)
    return output_img



    return normal_clone
def img_add(src_img, main_img, box, tree):
    global Record

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

        overlap = dx*dy
        box_area = (xmax-xmin)*(ymax-ymin)
        if overlap >= box_area*0.6:
            root.remove(box)
        else:
            Id = NametoId.get(box.find("name").text, -1)
            Record[Id] += 1


    if actual_count!=box_count:
        print(actual_count, box_count)
    root.append(new_box)
    
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
    img = img_clone(src_img, main_img, box, tree)
    #img = img_add(src_img, main_img, box, tree)
    if img is not None:
        global count 
        count += 1
        xml_path = os.path.join(output_annotation_path, '{:0>6d}'.format(count)+'.xml')
        jpg_path = os.path.join(output_Image_path, '{:0>6d}'.format(count)+'.jpg')

        tree.write(xml_path)
        cv2.imwrite(jpg_path, img)


data_folder_path = os.path.join('.', 'datalab-2021-cup2-object-detection')
train_annotation_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'Annotations')
train_Image_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'JPEGImages')
output_annotation_path = os.path.join('.', 'output', 'Annotations')
output_Image_path = os.path.join('.', 'output', 'JPEGImages')
box_list = []
box_dict = {}

Record = [0]*20 
box_pool = pd.DataFrame(columns=['target', 'name', 'Id', 'pose', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax'])
img_pool = pd.DataFrame(columns = [i for i in range(20)])
img_pool = pd.DataFrame(columns = [i for i in range(20)])
def balance_pick():
    global Record, box_pool, img_pool
    min_count, max_count = min(Record), max(Record)
    max1, max2, max3, max4, max5 = sorted(Record, reverse=True)[:5]
    if min_count*1.2 > max_count: # balanced
        return None, None
    else: # inbalanced
        min_class, max_class = Record.index(min_count), Record.index(max_count)
        max1_class = Record.index(max1)
        max2_class = Record.index(max2)
        max3_class = Record.index(max3)
        max4_class = Record.index(max4)
        max5_class = Record.index(max5)
        pick_box = box_pool[box_pool['Id']==min_class].sample(n=1).to_dict('records')[0]
        main_img = img_pool[(img_pool[max1_class]==0) & (img_pool[max2_class]==0) & (img_pool[max3_class]==0) & (img_pool[max4_class]==0) & (img_pool[max5_class]==0)].sample(n=1).index[0]
        return pick_box, main_img


def main():
    #target_list = [os.path.splitext(i)[0] for i in os.listdir(train_annotation_path)]
    target_list = [i.split('.')[0] for i in os.listdir(train_annotation_path)[:100]]
    global box_list, Record
    #global box_pool
    # read all box in train dataset
    #global Record

    for target in tqdm(target_list):
        
        #target_Image_path = os.path.join(train_Image_path, target + '.jpg')
        content = read_content(target)
        box_list += content
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output\JPEGImages'):
        os.mkdir('output\JPEGImages')
    if not os.path.exists('output\Annotations'):
        os.mkdir('output\Annotations')
    
    print(box_pool.head(5))
    print(Record)
    #print(img_pool)
    #return
    #for box in tqdm(box_list):
    #    target_main = np.random.choice(target_list)
    #    copy_paste(box, target_main)
    #return
    pick, target_main = balance_pick()
    
    while pick != None:
        Record[pick['Id']] += 1
        print(Record)
        #target_main = np.random.choice(target_list)
        #copy_paste(box_pool.iloc[14].to_dict(),'000026')
        copy_paste(pick, target_main)
        pick, target_main = balance_pick()


if __name__=='__main__':
    main()
    