import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

class DataSetLoader:
    def __init__(self, data_dir: str):

        self.data_dir = data_dir
        self.ann_dir = os.path.join(data_dir, "Annotations")
        self.img_dir = os.path.join(data_dir, "JPEGImages")
        
        if not os.path.exists(self.ann_dir) or not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"'Annotations' and 'JPEGImages' are not present in : {data_dir}")

    def load_dataset(self, max_images: int = None) -> List[Tuple[np.ndarray, Dict]]:

        xml_files= []
        for f in os.listdir(self.ann_dir):
            if f.endswith(".xml"):  
                xml_files.append(f)
        xml_files.sort()
        
        if max_images:
            xml_files = xml_files[:max_images]
            
        dataset= []
        print(f"Loading {len(xml_files)} images from {self.data_dir}...")
        
        for xml_file in xml_files:
            try:
                xml_path = os.path.join(self.ann_dir, xml_file)
                ann= self.parse_xml(xml_path)
                img= self.load_image_file(ann['filename'])
                dataset.append((img, ann))
            except Exception as e:
                print(f"Skipping {xml_file}: {e}")
                
        return dataset

    def parse_xml(self, xml_path: str) -> Dict:
        
        tree= ET.parse(xml_path)
        root= tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
              
            xmin= int(float(bndbox.find('xmin').text))-1
            ymin= int(float(bndbox.find('ymin').text))-1
            xmax= int(float(bndbox.find('xmax').text))-1
            ymax= int(float(bndbox.find('ymax').text))-1
            
            objects.append({
                'name': obj.find('name').text,
                'bbox': (xmin, ymin, xmax, ymax) 
            })
            
        return {'filename': root.find('filename').text, 'objects': objects}
    
    def load_image_file(self, filename: str) -> np.ndarray:
        
        if not filename.lower().endswith((".jpg", ".jpeg")):
            raise ValueError("file type not supported, only .jpg/.jpeg are supported.")

        path = os.path.join(self.img_dir, filename)
        img = cv2.imread(path)
        if img is None:
            base, ext = os.path.splitext(path)
            img = cv2.imread(base + ext.upper())

        if img is None:
            raise FileNotFoundError(f"Unable to read image: {filename}")

        return img

    