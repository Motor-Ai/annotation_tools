import os
import tkinter as tk
from PIL import Image, ImageTk
import re
import argparse
import cv2
import re
import time
import json
import torch
import numpy as np
import xml.etree.ElementTree as ET
from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert


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
        logit = float(obj.find('score').text)
        # score = float(obj.find('score').text)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        logits.append(logit)
        # logits.append(score)
    # Convert the list to a numpy array
    # logits = np.ones(len(boxes))
    boxes_array = torch.from_numpy(np.array(boxes))
    return boxes_array, labels, logits

def write_bbox_to_xml(bbox_annotation, labels, logits, output_file, filename, source, img_dims):
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

class SimpleApp:
    def __init__(self, root, image_folder, label_folder):
        self.root = root
        self.root.title("Image Click App")
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = self.get_image_files()
        self.label_files = get_sorted_files(label_folder, 'xml')

        self.current_image_index = 0
        self.current_annotation = {'boxes': None, 'phrases': None, 'logits': None}
        w, h = Image.open(self.image_files[0]).size
        self.width = w
        self.height = h
        self.play_speed = 25 
        self.autoplay = False
 
        self.load_image()

        self.canvas = tk.Canvas(self.root, width=self.photo.width(), height=self.photo.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.canvas.bind("<Button-1>", self.handle_click)
        self.root.bind('a', self.start_autoplay)

        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)
        self.canvas.bind("<ButtonPress-3>", self.handle_right_click)
        self.canvas.bind("<ButtonRelease-3>", self.handle_right_release)

    def get_image_files(self):
        image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        return get_sorted_files(self.image_folder, image_extensions)

        # return [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith(image_extensions)]

    def load_image(self):
        # self.image = Image.open(self.image_files[self.current_image])
        # self.photo = ImageTk.PhotoImage(self.image)
        image_source, image = load_image(self.image_files[self.current_image_index])
        boxes, phrases, logits = get_annotation_from_xml(self.label_files[self.current_image_index]) 
        print(f'boxes for file {self.label_files[self.current_image_index]} = \n {boxes}')
        
        if len(boxes) > 0:
            # print(f'boxes! = {boxes}')
            # temp = boxes * torch.Tensor([self.width, self.height, self.width, self.height])
            # self.current_annotation['boxes'] = box_convert(boxes=temp, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            self.current_annotation['boxes'] = boxes
            self.current_annotation['phrases'] = phrases
            self.current_annotation['logits'] = logits
            print(f'phrases = {phrases}')
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            self.current_image = annotated_frame
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGBA)
        else:
            self.current_image = image_source

        # mask = cv2.imread(, cv2.IMREAD_GRAYSCALE)
        # mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        # self.current_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
        # self.current_image = cv2.resize(self.current_image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        self.current_image = Image.fromarray(self.current_image)
        # self.current_image.thumbnail((1090, 890))
        self.photo = ImageTk.PhotoImage(image=self.current_image)

        self.root.title(f'{os.path.basename(self.image_files[self.current_image_index])} - {self.current_image_index + 1}/{len(self.image_files)}')
        if self.autoplay:
            self.root.after(self.play_speed, self.next_image)

        # self.root.geometry(f"{self.photo.width()}x{self.photo.height()}")
    def handle_right_click(self, event):
        self.start_x, self.start_y = event.x, event.y
        print(f'click coords = ({self.start_x}, {self.start_y})')


    def handle_right_release(self, event):
        end_x, end_y = event.x, event.y
        box = (self.start_x, self.start_y, end_x, end_y)
        box = box_convert(boxes=torch.Tensor([box]),  in_fmt="xyxy", out_fmt="cxcywh")
        box_normalized = box*torch.Tensor([1/self.width, 1/self.height, 1/self.width, 1/self.height])

        print(f'click coords = ({end_x}, {end_y})')

        boxes = self.current_annotation['boxes']
        phrases = self.current_annotation['phrases']
        logits = self.current_annotation['logits']

        if boxes is None:
            boxes = box_normalized
            phrases = ['traffic sign']
            logits = [1.0]
        else:
            print(f'boxes = {boxes}')
            print(f'box = {box}')
            print(f'box_norm = {box_normalized}')

            boxes = torch.cat((boxes, box_normalized))
        
            phrases.append('traffic sign')
            logits.append(1.0)

        rescaled_box = box_normalized*torch.Tensor([self.width, self.height, self.width, self.height])
        rescaled_box = (rescaled_box[0][0], rescaled_box[0][1], rescaled_box[0][2], rescaled_box[0][3])
        print(f'rescaled_box = {rescaled_box}')
        print(f'box = {box}')

        # boxes_cxcywh = box_convert(boxes=boxes, out_fmt="cxcywh", in_fmt="xyxy").numpy()

        write_bbox_to_xml(bbox_annotation=boxes, 
                    labels=phrases, 
                    logits=logits, 
                    output_file=self.label_files[self.current_image_index], 
                    filename=self.image_files[self.current_image_index], 
                    source=self.image_folder, 
                    img_dims=(self.width, self.height, 3))
        
        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)



    def handle_click(self, event):
        x, y = event.x, event.y
        print(f'click coords = ({x}, {y})')
        size = 2
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red", width=2)
        boxes = self.current_annotation['boxes']
        phrases = self.current_annotation['phrases']
        logits = self.current_annotation['logits']
        clicked_square_index = None 
        if boxes is None or len(boxes) == 0:
            return
        boxes_scaled = box_convert(boxes=boxes*torch.Tensor([self.width, self.height, self.width, self.height]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # temp = boxes * torch.Tensor([self.width, self.height, self.width, self.height])
        # self.current_annotation['boxes'] = box_convert(boxes=temp, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        for i, square in enumerate(boxes_scaled):
            xmin, ymin, xmax, ymax = square
            if xmin <= x <= xmax and ymin <= y <= ymax:
                clicked_square_index = i
                break

        if clicked_square_index is not None:
            print(f"Clicked square index: {clicked_square_index}")

            good_labels = np.array(list(set(range(len(boxes))) - set([clicked_square_index])), dtype=np.int8)

            print(f'before deletion {boxes}')
            print(f'good labels = {good_labels}')

            boxes = boxes[good_labels]
            print(f'after deletion {boxes}')
            
            phrases = [phrases[i] for i in good_labels]
            # print(type(logits))
            # print(type(good_labels))
            logits = [logits[i] for i in good_labels]

            write_bbox_to_xml(bbox_annotation=boxes, 
                              labels=phrases, 
                              logits=logits, 
                              output_file=self.label_files[self.current_image_index], 
                              filename=self.image_files[self.current_image_index], 
                              source=self.image_folder, 
                              img_dims=(self.width, self.height, 3))
            
            print(f'updated annotations for {self.label_files[self.current_image_index]}')

        else:
            print("Clicked outside the squares")

        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def previous_image(self, event):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def next_image(self, event=None):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def start_autoplay(self, event=None):
        self.autoplay = not self.autoplay
        self.next_image()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--session-name', help='name of the recording session', required=True)
    parser.add_argument('-t', '--session-type', help='type of the recording session (train or val or test)', required=True)
    parser.add_argument('-p', '--play-speed', help='speed of data playback', default=50)

    args = vars(parser.parse_args())    
    root_path = os.path.join(os.path.expanduser('~'), 'datasets', args['session_type'], args['session_name'])

    image_folder = os.path.join(root_path, 'images')
    label_folder = os.path.join(root_path, 'labels')
    root = tk.Tk()
    app = SimpleApp(root, image_folder, label_folder)
    root.mainloop()