import xml.etree.ElementTree as ET

def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    
    for box in root.iter('object'):
        output_box = {
            'target':root.find('filename').text.split('.')[0],
            'name':box.find('name').text,
            'pose':box.find('pose').text,
            'truncated':box.find('truncated').text,
            'difficult':box.find('difficult').text,
            'xmin':int(box.find("bndbox/xmin").text),
            'ymin':int(box.find("bndbox/ymin").text),
            'xmax':int(box.find("bndbox/xmax").text),
            'ymax':int(box.find("bndbox/ymax").text)
        }
        list_with_all_boxes.append(output_box)
    return list_with_all_boxes