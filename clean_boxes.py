import os
import argparse

import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import albumentations
import xml.etree.ElementTree as ET
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate

from utils import write_bbox_to_xml, get_annotation_from_xml, get_sorted_files, change_label_in_xml, rescale_bounding_box, print_color


class SimpleApp:
    def __init__(self, root, image_folder, label_folder):
        self.root = root
        self.root.title("Image Click App")
        # Get the screen dimensions
        root.update_idletasks()
        self.screen_width = 1920
        self.screen_height = 1080
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
        self.start_x, self.start_y = None, None
        
        self.scale_width = self.width/1920
        self.scale_height = self.height/1080
        
        self.zoomed_region = None
        self.zoom_level = 1
        
        self.labels = ['Pedestrian', 'Stop', 'Speed_Limit_30', 'Speed_Limit_50']

 
        self.load_image()

        self.canvas = tk.Canvas(self.root, width=self.screen_width, height = self.screen_height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.canvas.bind("<Button-1>", self.handle_click)
        self.root.bind('a', self.start_autoplay)
        self.root.bind('s', self.stop_autoplay)
        self.root.bind('d', self.delete_all)
        self.root.bind('q', self.start_autoplay_with_delete)

        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)
        self.canvas.bind("<ButtonPress-3>", self.handle_right_click)
        self.canvas.bind("<ButtonRelease-3>", self.handle_right_release)
        self.canvas.bind("<ButtonPress-2>", self.handle_wheel_click)
        self.canvas.bind("<ButtonRelease-2>", self.handle_wheel_release)
        self.canvas.bind("<B3-Motion>", self.handle_motion)
    
        # Create the buttons on the canvas
        self.buttons = []
        for i, label in enumerate(self.labels):
            button = tk.Button(self.canvas, text=label, command=lambda idx=i: self.set_label_for_images(idx))
            button_window = self.canvas.create_window(10, 10 + i * 30, anchor=tk.NW, window=button)
            self.buttons.append((button, button_window))
            


    def set_label_for_images(self, label_index):

        self.current_label = self.labels[label_index]
        print_color(f"Changed the annotation label to {self.current_label}", 'green')

    def get_image_files(self):
        image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        return get_sorted_files(self.image_folder, image_extensions)

        # return [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.lower().endswith(image_extensions)]

    def load_image(self):
        image_source, image = load_image(self.image_files[self.current_image_index])
        
        boxes, phrases, logits = get_annotation_from_xml(self.label_files[self.current_image_index])
        
        boxes = torch.as_tensor([rescale_bounding_box(self.width, self.height, self.screen_width, self.screen_height, bbox) for bbox in boxes])
        resized_image = cv2.resize(image_source, (self.screen_width, self.screen_height), interpolation = cv2.INTER_AREA)

        print_color(f'Boxes for file {self.label_files[self.current_image_index]} = \n {boxes}', 'green')

        self.current_annotation['boxes'] = boxes
        self.current_annotation['phrases'] = phrases
        self.current_annotation['logits'] = logits
        if len(boxes) > 0:
            annotated_frame = annotate(image_source=resized_image, boxes=boxes, logits=logits, phrases=phrases, bbox_format = "xyxy")
            self.current_image = annotated_frame
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGBA)
        else:
            self.current_image = resized_image

        self.current_image = Image.fromarray(self.current_image)
        self.photo = ImageTk.PhotoImage(image=self.current_image)

        self.root.title(f'{os.path.basename(self.image_files[self.current_image_index])} - {self.current_image_index + 1}/{len(self.image_files)}')
        if self.autoplay:
            self.root.after(self.play_speed, self.next_image)

    def handle_right_click(self, event):
        self.start_x, self.start_y = event.x, event.y

    def delete_all(self, event=None):
        boxes = torch.Tensor([])
        phrases = []
        logits = []
        write_bbox_to_xml(bbox_annotation=boxes, 
            labels=phrases, 
            logits=logits, 
            output_file=self.label_files[self.current_image_index], 
            filename=self.image_files[self.current_image_index], 
            source=self.image_folder, 
            img_dims=(self.width, self.height, 3))
                
        self.current_annotation['boxes'] = boxes
        self.current_annotation['phrases'] = phrases
        self.current_annotation['logits'] = logits
        
        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def handle_right_release(self, event):
        end_x, end_y = event.x, event.y
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            print_color("Can't create boxes for a single point!", "red")
            self.canvas.delete("temp_square")
            return
        xmin = min(self.start_x, end_x)
        ymin = min(self.start_y, end_y)
        xmax = max(self.start_x, end_x)
        ymax = max(self.start_y, end_y)
        
        # box = (self.start_x, self.start_y, end_x, end_y)
        box = torch.Tensor([xmin, ymin, xmax, ymax])
        
        if self.zoomed_region == None:
        
            # box = box_convert(boxes=torch.Tensor([box]),  in_fmt="xyxy", out_fmt="cxcywh")
            box_normalized = torch.unsqueeze(box*torch.Tensor([1/self.width, 1/self.height, 1/self.width, 1/self.height]), 0)
            box_normalized = [[box[0]*self.scale_width, box[1]*self.scale_height, box[2]*self.scale_width, box[3]*self.scale_height] for box in box_normalized]
            box_normalized = torch.as_tensor(box_normalized)
            
        else:
            zoomed_width = self.zoomed_region[2] - self.zoomed_region[0] 
            zoomed_height = self.zoomed_region[3] - self.zoomed_region[1] 
            
            
            scale_factor_width = zoomed_width/self.width
            scale_factor_height = zoomed_height/self.height
            
            offset_x = self.zoomed_region[0]
            offset_y = self.zoomed_region[1]
            
            old_xmin = box[0] * scale_factor_width + offset_x
            old_ymin = box[1] * scale_factor_height + offset_y
            old_xmax = box[2] * scale_factor_width + offset_x
            old_ymax = box[3] * scale_factor_height + offset_y
            
            box = [old_xmin*self.scale_width, old_ymin*self.scale_height, old_xmax*self.scale_width, old_ymax*self.scale_height]
    
            box = torch.Tensor([old_xmin, old_ymin, old_xmax, old_ymax])
            
            # box = box_convert(boxes=torch.Tensor([box]),  in_fmt="xyxy", out_fmt="cxcywh")
            box_normalized = torch.unsqueeze(box*torch.Tensor([1/self.width, 1/self.height, 1/self.width, 1/self.height]), 0)
        
        
        boxes = self.current_annotation['boxes']
        phrases = self.current_annotation['phrases']
        logits = self.current_annotation['logits']


        if boxes is None:
            boxes = box_normalized
            phrases = [self.current_label]
            logits = [1.0]
        else:
            boxes = torch.cat((boxes, box_normalized), dim = 0)
        
            phrases.append(self.current_label)
            logits.append(1.0)

        # boxes_cxcywh = box_convert(boxes=boxes, out_fmt="cxcywh", in_fmt="xyxy").numpy()

        write_bbox_to_xml(bbox_annotation=boxes, 
                    labels=phrases, 
                    logits=logits, 
                    output_file=self.label_files[self.current_image_index], 
                    filename=self.image_files[self.current_image_index], 
                    source=self.image_folder, 
                    img_dims=(self.width, self.height, 3),
                    bbox_format='xyxy')
        
        self.current_annotation['boxes'] = boxes
        self.current_annotation['phrases'] = phrases
        self.current_annotation['logits'] = logits

        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.delete("temp_square")
        self.start_x, self.start_y = None, None
        self.zoom_level = 1.0
        self.zoomed_region = None

    def handle_motion(self, event):
        if self.start_x and self.start_y:
            self.canvas.delete("temp_square")
            temp_square = (self.start_x, self.start_y, event.x, event.y)
            self.canvas.create_rectangle(temp_square, outline="blue", width=2, tags="temp_square")

    def handle_click(self, event):
        x, y = event.x*self.scale_width, event.y*self.scale_height
        size = 2
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="red", width=2)
        boxes = self.current_annotation['boxes']
        phrases = self.current_annotation['phrases']
        logits = self.current_annotation['logits']
        clicked_square_index = None 
        if boxes is None or len(boxes) == 0:
            return
        boxes_scaled = boxes*torch.Tensor([self.width, self.height, self.width, self.height])

        for i, square in enumerate(boxes_scaled):
            xmin, ymin, xmax, ymax = square
            if xmin <= x <= xmax and ymin <= y <= ymax:
                clicked_square_index = i
                break

        if clicked_square_index is not None:
            print_color(f"Clicked square index: {clicked_square_index}", 'green')

            good_labels = np.array(list(set(range(len(boxes))) - set([clicked_square_index])), dtype=np.int8)

            boxes = boxes[good_labels]
            
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
                              img_dims=(self.width, self.height, 3),
                              bbox_format='xyxy')
            
            self.current_annotation['boxes'] = boxes
            self.current_annotation['phrases'] = phrases
            self.current_annotation['logits'] = logits
            
            print_color(f'Updated annotations for {self.label_files[self.current_image_index]}', 'green')

        else:
            print_color("Clicked outside the squares", 'red')
            return

        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def previous_image(self, event):
        if self.current_image_index > 0:
            change_label_in_xml(self.label_files[self.current_image_index], self.current_label)
            self.current_image_index -= 1
            self.load_image()
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def next_image(self, event=None):
        if self.current_image_index < len(self.image_files) - 1:
            change_label_in_xml(self.label_files[self.current_image_index], self.current_label)
            self.current_image_index += 1
            self.load_image()
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def start_autoplay(self, event=None):
        self.autoplay = not self.autoplay
        self.next_image()
        
    def start_autoplay_with_delete(self, event):
        boxes = torch.Tensor([])
        phrases = []
        logits = []
        write_bbox_to_xml(bbox_annotation=boxes, 
            labels=phrases, 
            logits=logits, 
            output_file=self.label_files[self.current_image_index], 
            filename=self.image_files[self.current_image_index], 
            source=self.image_folder, 
            img_dims=(self.width, self.height, 3))
                
        self.current_annotation['boxes'] = boxes
        self.current_annotation['phrases'] = phrases
        self.current_annotation['logits'] = logits
        
        self.next_image()

    def stop_autoplay(self, event=None):
        self.autoplay = False
        self.load_image()
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        # self.next_image()

    def handle_wheel_click(self, event):
        self.zoom_start_x, self.zoom_start_y = event.x, event.y

    def handle_wheel_release(self, event):
        zoom_end_x = int(event.x / self.zoom_level)
        zoom_end_y = int(event.y / self.zoom_level)
        self.zoomed_region = (self.zoom_start_x*self.scale_width, self.zoom_start_y*self.scale_height, zoom_end_x*self.scale_width, zoom_end_y*self.scale_height)
        x1, y1, x2, y2 = self.zoomed_region
        img = Image.open(self.image_files[self.current_image_index])
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((self.width, self.height))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.buttons = []
        for i, label in enumerate(self.labels):
            button = tk.Button(self.canvas, text=label, command=lambda idx=i: self.set_label_for_images(idx))
            button_window = self.canvas.create_window(10, 10 + i * 30, anchor=tk.NW, window=button)
            self.buttons.append((button, button_window))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_folder', '--images_folder', help='name of the recording session', required=True)
    parser.add_argument('-p', '--play-speed', help='speed of data playback', default=50)

    args = vars(parser.parse_args())    
    root_path = args['images_folder']

    image_folder = os.path.join(root_path, 'images')
    label_folder = os.path.join(root_path, 'labels')
    root = tk.Tk()
    app = SimpleApp(root, image_folder, label_folder)
    root.mainloop()