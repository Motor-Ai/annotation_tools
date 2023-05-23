import os
import glob
from tkinter import Label, Tk, LEFT
from PIL import Image, ImageTk
import shutil
import argparse
import cv2
import re
import time
import json


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

def file_idx_in_set(file_path, file_indices):
    # file = os.path.split(file_path)[1]
    idx = get_image_idx(file_path, full_path=True)
    return idx in file_indices


class ImageViewer:
    def __init__(self, master, image_folder, mask_folder, json_file_path, play_speed=50):
        self.master = master
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        # self.image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        self.image_files = get_sorted_files(image_folder, 'jpg')
        self.image_files.extend(get_sorted_files(image_folder, 'png'))
        self.mask_files = get_sorted_files(mask_folder, 'jpg')
        self.mask_files.extend(get_sorted_files(mask_folder, 'png'))
        self.all_image_files = list(self.image_files)
        self.all_mask_files = list(self.mask_files)
        self.json_file_path = json_file_path
        self.json_file = None
        # if json provided then load only images with good masks
        if os.path.exists(json_file_path):
            input('json file already exists, will be updated, continue? (y/n): ')    
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                self.json_file = data
            bad_indices = data['bad_mask_indices']            
            good_image_files = list(filter(lambda x: not file_idx_in_set(x, bad_indices), self.image_files))
            good_mask_files = list(filter(lambda x: not file_idx_in_set(x, bad_indices), self.mask_files))


            self.image_files = good_image_files
            self.mask_files = good_mask_files

        self.current_image_index = 0
        self.auto_select_flag = False
        self.autoplay = False
        self.selected_images = set()
        self.width, self.height = 1100, 900
        self.play_speed = play_speed  

        self.display_image()
    
        self.master.bind('<Left>', self.prev_image)
        self.master.bind('s', self.auto_select)
        self.master.bind('a', self.start_autoplay)

        self.master.bind('<Right>', self.next_image)
        self.master.bind('<space>', self.select_image)
        self.master.bind('<Delete>', self.save_selections)

        self.commands_label = Label(self.master, text=
                                "Commands: Left Arrow |  Right Arrow | Space (select) | Delete (remove) | s key (start/stop sequence) | a key (start/stop autoplay)",
                                justify=LEFT)
        self.commands_label.pack(side=LEFT)

    def get_bad_mask_files(self):
        return list(filter(lambda x: get_number(x) in self.selected_images, self.mask_files))

    
    def display_image(self):
        # mask_path = f"{image_folder}_masks"
        # mask_path = os.path.join(mask_path,  os.path.split(self.image_files[self.current_image_index])[1])

        image = cv2.imread(self.image_files[self.current_image_index])
        mask = cv2.imread(self.mask_files[self.current_image_index], cv2.IMREAD_GRAYSCALE)
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        self.current_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

        self.current_image = cv2.resize(self.current_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGBA)

        self.current_image = Image.fromarray(self.current_image)
        # self.current_image.thumbnail((1090, 890))

        self.photo = ImageTk.PhotoImage(image=self.current_image)


        if hasattr(self, 'label'):
            self.label.config(image=self.photo)
        else:
            self.label = Label(self.master, image=self.photo)
            self.label.pack()

        if self.current_image_index in self.selected_images:
            self.label.config(bg='red')
        else:
            self.label.config(bg='white')


        self.master.title(f'{os.path.basename(self.image_files[self.current_image_index])} - {self.current_image_index + 1}/{len(self.image_files)}')
        if self.autoplay:
            self.master.after(self.play_speed, self.next_image)


    def prev_image(self, event=None):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def next_image(self, event=None):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_image()
        if self.auto_select_flag:
            self.select_image()
    
    def auto_select(self, event=None):
        self.auto_select_flag = not self.auto_select_flag
        self.select_image()

    def start_autoplay(self, event):
        self.autoplay = not self.autoplay
        self.next_image()

    def stop_autoplay(self, event):
        self.autoplay = False

    def select_image(self, event=None):
        if self.current_image_index in self.selected_images:
            self.selected_images.remove(self.current_image_index)
            self.label.config(bg='white')
        else:
            self.selected_images.add(self.current_image_index)
            self.label.config(bg='red')

    def get_selected_file_indices(self):
        file_indices = set()
        for index in self.selected_images:
            file_index = get_image_idx(self.image_files[index], full_path=True)
            file_indices = file_indices.union([file_index])
        return file_indices
    
    def get_all_file_indices(self):
        return set(map(get_image_idx, self.all_image_files))
    
    def get_not_selected_file_indices(self):
        not_selected_indices = set()
        for file in self.image_files:
            idx = get_image_idx(file)
            if idx in self.selected_images:
                continue
            not_selected_indices = not_selected_indices.union([idx])
        return not_selected_indices

    def save_selections(self, event=None):
        # for index in sorted(self.selected_images, reverse=True):
            # new_file = os.path.split(self.image_files[index])[1]
            # shutil.move(self.image_files[index], os.path.join(self.discarded_images_path, new_file))
            # shutil.move(self.mask_files[index], os.path.join(self.discarded_masks_path, new_file))
            # del self.image_files[index]
        with open(self.json_file_path, 'w') as f:
            bad_file_indices = self.get_selected_file_indices()
            if self.json_file is not None:
                bad_file_indices = bad_file_indices.union(set(self.json_file['bad_mask_indices']))

            good_file_indices = self.get_all_file_indices().difference(bad_file_indices)
            # print(f'good file_indices after difference {good_file_indices}')
            json.dump({'good_mask_indices': list(good_file_indices), 
                        'bad_mask_indices': list(bad_file_indices), 
                        'mask_folder': self.mask_folder, 
                        'image_folder': self.image_folder}, f)

        self.selected_images.clear()
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = len(self.image_files) - 1

        self.display_image()
        self.master.destroy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This script gets a path to folder containing images with the format <idx>.<extension>
    and a .txt file with indices of images to remove from the folder. The indices can be single number or ranges (e.g. 1 or 5-10)
    after removing the images it adds the _good  sufffix to the original folder""")
    # parser.add_argument('-i','--image-folder', help='path to folder with images', required=True)
    # parser.add_argument('-m','--mask-folder', help='path to folder with corresponding masks', required=True)
    # parser.add_argument('-j','--json-file', help='path to json file with indices', required=True)
    parser.add_argument('-s', '--session-name', help='name of the recording session', required=True)
    parser.add_argument('-t', '--session-type', help='type of the recording session (train or val or test)', required=True)
    parser.add_argument('-c', '--camera-index', help='index of the camera that produced the data', required=True)
    parser.add_argument('-p', '--play-speed', help='speed of data playback', default=50)

    args = vars(parser.parse_args())

    
    root_path = os.path.join(os.path.expanduser('~'), 'datasets', args['session_type'], args['session_name'], f'cam{args["camera_index"]}')

    # image_folder = os.path.join(root_path, args['image_folder'])
    # mask_folder = os.path.join(root_path, args['mask_folder'])
    # json_file = os.path.join(root_path, args['json_file'])

    image_folder = os.path.join(root_path, 'images')
    mask_folder = os.path.join(root_path, 'labels')
    json_file = os.path.join(root_path, 'labels_info.json')
    play_speed = int(args['play_speed'])

    # mask_folder = os.path.join(root_path, args['mask_folder'])

    # json_file = os.path.join(root_path, args['json_file'])


    # image_folder = '/home/sam/work/python_scripts/2d_bbox_demo' # Replace with the path to your folder containing the images
    root = Tk()
    ImageViewer(root, image_folder, mask_folder, json_file)
    root.mainloop()
