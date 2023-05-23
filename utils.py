import os
import re
from typing import List


import cv2
import torch
import numpy as np
import supervision as sv
import xml.etree.ElementTree as ET
from torchvision.ops import box_convert


def write_bbox_to_xml(bbox_annotation, labels, logits, output_file, filename=None, source=None, img_dims=None, box_threshold=None, text_threshold=None, text_prompt=None,
                      cleaned=None, bbox_format = 'cxcywh'):
    # Parse the existing XML file
    tree = ET.parse(output_file)
    annotation = tree.getroot()

    # Update the variables with new values
    if filename is not None:
        fn = annotation.find("filename")
        fn.text = filename
    if source is not None:
        src = annotation.find("source")
        src.text = source
    if img_dims is not None:
        size = annotation.find("size")
        width_ = size.find("width")
        width_.text = str(img_dims[0])
        height_ = size.find("height")
        height_.text = str(img_dims[1])
        depth_ = size.find("depth")
        depth_.text = str(img_dims[2])

    # Remove existing object elements
    for obj in annotation.findall("object"):
            annotation.remove(obj)

    # Add new object elements
    for i in range(bbox_annotation.shape[0]):
        if labels[i] == '':
            continue
        
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = labels[i]
        score = ET.SubElement(obj, "score")
        score.text = str(float(logits[i]))

        if box_threshold is not None:
            box_threshold_ = ET.SubElement(obj, "box_threshold")
            box_threshold_.text = str(box_threshold)
        if text_threshold is not None:
            text_threshold_ = ET.SubElement(obj, "text_threshold")
            text_threshold_.text = str(text_threshold)
        if text_prompt is not None:
            text_prompt_ = ET.SubElement(obj, "text_prompt")
            text_prompt_.text = str(text_prompt)
        if cleaned is not None:
            cleaned_ = ET.SubElement(obj, "cleaned")
            cleaned_.text = str(cleaned)
            
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = '0'
        occluded = ET.SubElement(obj, "occluded")
        occluded.text = str(0)
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmax = ET.SubElement(bndbox, "xmax")
        
        if bbox_format=='cxcywh':
            xyxy_bbox =  box_convert(boxes=bbox_annotation[i], in_fmt="cxcywh", out_fmt="xyxy")
        else:
            xyxy_bbox = bbox_annotation[i]
            print("the xyxy box: ", xyxy_bbox)
        # xyxy_bbox =  box_convert(boxes=bbox_annotation[i], in_fmt="cxcywh", out_fmt="xyxy")
            
        xmin.text = str(xyxy_bbox[0].item())
        xmax.text = str(xyxy_bbox[2].item())
        ymin = ET.SubElement(bndbox, "ymin")
        ymax = ET.SubElement(bndbox, "ymax")
        ymin.text = str(xyxy_bbox[1].item())
        ymax.text = str(xyxy_bbox[3].item())
        area = bbox_annotation[i][2].item() * bbox_annotation[i][3].item() * img_dims[0] * img_dims[1]
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = str(area)

    # Write the updated XML to the same file
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


def change_label_in_xml(xml_file, new_label):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Update the label for each object
    for obj in root.findall("object"):
        name = obj.find("name")
        name.text = new_label

    # Write the updated XML to the same file
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)


def get_number(file):
    n = re.findall(r'\d+', file)[0]
    return int(n)

def get_image_idx(file, full_path=True):
    if full_path:
        file = os.path.split(file)[1]
    return get_number(file)   

def get_sorted_files(folder, format):
  list_of_files = [f for f in os.listdir(folder) if f.endswith(format)]
  list_of_files.sort(key=get_number)
  result = [os.path.join(folder, file) for file in list_of_files]
  return result


def convert_coordinates(bounding_box,  new_width, new_height, original_width, original_height):
    xmin, ymin, xmax, ymax = bounding_box
        # Calculate the aspect ratios
    original_aspect_ratio = original_width / original_height
    new_aspect_ratio = new_width / new_height
    
    # Determine which dimension (width or height) needs to be scaled more
    if new_aspect_ratio > original_aspect_ratio:
        # Scale based on height
        scale_factor = original_height / new_height
        scaled_width = new_width * scale_factor
        offset = (original_width - scaled_width) / 2
        old_xmin = xmin * scale_factor + offset
        old_xmax = xmax * scale_factor + offset
        old_ymin = ymin * scale_factor
        old_ymax = ymax * scale_factor
    else:
        # Scale based on width
        scale_factor = original_width / new_width
        scaled_height = new_height * scale_factor
        offset = (original_height - scaled_height) / 2
        old_xmin = xmin * scale_factor
        old_xmax = xmax * scale_factor
        old_ymin = ymin * scale_factor + offset
        old_ymax = ymax * scale_factor + offset
    
    # Return the converted coordinates
    return old_xmin, old_ymin, old_xmax, old_ymax


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], scaled: bool = False, format: str = "cxcywh") -> np.ndarray:
    h, w, _ = image_source.shape
    if not scaled:
        boxes = boxes * torch.Tensor([w, h, w, h])
    if format == "cxcywh":
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame