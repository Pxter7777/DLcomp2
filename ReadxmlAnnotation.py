from math import trunc
import xml.etree.ElementTree as ET
import os.path

def read_content(xml_file: str):

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
       
        
        box = {
            'name':name,
            'pose':pose,
            'truncated':truncated,
            'difficult':difficult,
            'xmin':xmin,
            'ymin':ymin,
            'xmax':xmax,
            'ymax':ymax
        }
        #list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(box)

    return filename, list_with_all_boxes

def main():
    data_folder_path = os.path.join('.', 'datalab-2021-cup2-object-detection')
    train_annotation_path = os.path.join(data_folder_path, 'VOCdevkit_train', 'VOC2007', 'Annotations')
    target_xml_path = os.path.join(train_annotation_path, '000005.xml')
    print(target_xml_path)
    name, boxes = read_content(target_xml_path)
    print(name, boxes)

if __name__=='__main__':
    main()
    