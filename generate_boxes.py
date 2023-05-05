import numpy as np
import xml.etree.ElementTree as ET
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
from tqdm import tqdm
import argparse
import torch

def write_bbox_to_xml(bbox_annotation, labels, logits, output_file, filename, source, img_dims, box_threshold, text_threshold, text_prompt,
                      cleaned=0):
    # Create the XML structure
    # header
    annotation = ET.Element("annotation")
    fn = ET.SubElement(annotation, "filename")
    fn.text = filename
    src = ET.SubElement(annotation, "source")
    src.text = source
    size = ET.SubElement(annotation, "size")
    width_ =  ET.SubElement(size, "width")
    width_.text = str(img_dims[0])
    height_ =  ET.SubElement(size, "height")
    height_.text = str(img_dims[1])
    depth_ =  ET.SubElement(size, "depth")
    depth_.text = str(img_dims[2])
    # one object per bbox
    for i in range(bbox_annotation.shape[0]):
        # object
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = labels[i]
        score = ET.SubElement(obj, "score")
        score.text = str(float(logits[i]))

        box_threshold_ = ET.SubElement(obj, "box_threshold")
        box_threshold_.text = str(box_threshold)
        text_threshold_ = ET.SubElement(obj, "text_threshold")
        text_threshold_.text = str(text_threshold)
        text_prompt_ = ET.SubElement(obj, "text_prompt")
        text_prompt_.text = str(text_prompt)
        cleaned_ = ET.SubElement(obj, "cleaned")
        cleaned_.text = str(cleaned)
        
        pose = ET.SubElement(obj, "pose")
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = '0'
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = str(0)
        occluded =  ET.SubElement(obj, "occluded")
        occluded.text = str(0)
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmax = ET.SubElement(bndbox, "xmax")
        xmin.text = str(float(bbox_annotation[i][0]))
        xmax.text = str(float(bbox_annotation[i][2]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymax = ET.SubElement(bndbox, "ymax")
        ymin.text = str(float(bbox_annotation[i][1]))
        ymax.text = str(float(bbox_annotation[i][3]))

    # Write the XML to a file
    tree = ET.ElementTree(annotation)
    with open(output_file, "wb") as f:
        f.write(b'<?xml version="1.0" ?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)



def get_annotation_from_xml(xml_path):
    # xml_path = '/home/sam/work/tasks/traffic_sign_detection/GroundingDINO/data/edge_case_traffic_sign.xml'
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize an empty list to store the values
    boxes = []
    labels = []
    logits = []

    # Iterate through the XML elements and extract the values
    for obj in root.findall('object'):
        label = obj.find('name').text
        score = float(obj.find('score').text)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        logits.append(score)

    # Convert the list to a numpy array
    boxes_array = torch.from_numpy(np.array(boxes))
    return boxes_array, labels, logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input folder')
    parser.add_argument('-o', help='output folder')
    parser.add_argument('-p', help='text prompt', default="traffic light.")
    args = vars(parser.parse_args())

    image_path = args['i']
    output_path = args['o']

    # image_path = '/home/sam/work/tasks/traffic_sign_detection/GroundingDINO/data'
    # output_path = '/home/sam/work/tasks/traffic_sign_detection/labels'
    TEXT_PROMPT = args['p']

    os.makedirs(output_path, exist_ok=True)
    weights_path = "/home/sam/work/annotation_tools/weights/groundingdino_swint_ogc.pth"
    model_path = "/home/sam/work/annotation_tools/models/GroundingDINO_SwinT_OGC.py"
    model = load_model(model_path, weights_path)
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.3
    # img_dims = 1900, 1200, 3

    for file in tqdm(os.listdir(image_path)):
        root_path, image_folder = os.path.split(image_path)
        img_filename = os.path.join(image_path, file)
        output_img_filename = os.path.join(output_path, file)
        label_filename = file.replace('.jpg', '.xml')
        label_path = os.path.join(output_path, label_filename)
        image_source, image = load_image(img_filename)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        img_dims = image_source.shape
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        write_bbox_to_xml(bbox_annotation=boxes, labels=phrases, logits=logits, 
                          output_file=label_path, filename=file, source=image_path, 
                          img_dims=img_dims, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD,
                          text_prompt=TEXT_PROMPT)
        
        # boxes, phrases, logits = get_annotation_from_xml(label_path)
        # cv2.imwrite(output_img_filename.replace('.jpg', '_label.jpg'), annotated_frame)

