import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from typing import Any, Dict, List
import torchvision
import time
import glob
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str):
  header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
  metadata = [header]
  for i, mask_data in enumerate(masks):
      mask = mask_data["segmentation"]
      filename = f"{i}.png"
      cv2.imwrite(os.path.join(path, filename), mask * 255)
      mask_metadata = [
          str(i),
          str(mask_data["area"]),
          *[str(x) for x in mask_data["bbox"]],
          *[str(x) for x in mask_data["point_coords"][0]],
          str(mask_data["predicted_iou"]),
          str(mask_data["stability_score"]),
          *[str(x) for x in mask_data["crop_box"]],
      ]
      row = ",".join(mask_metadata)
      metadata.append(row)
  metadata_path = os.path.join(path, "metadata.csv")
  with open(metadata_path, "w") as f:
      f.write("\n".join(metadata))

  return

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=400):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)  
    # Add unique numbers to each point
    for i, coord in enumerate(coords):
        ax.text(coord[0], coord[1], str(i), fontsize=15, weight='bold', color='black', ha='center', va='center')

def get_number(file):
  return int(file.split('.', 1)[0])

def get_sorted_files(folder):
  list_of_files = [f for f in os.listdir(folder) if f.endswith("png")]
  list_of_files.sort(key=get_number)
  return list_of_files


def get_road_indices_from_mask(mask, n_points):
  all_road_indices = np.argwhere(mask)
  candidate_road_indices = np.random.choice(len(all_road_indices), n_points, replace=False)
  candidate_indices = all_road_indices[candidate_road_indices]
  # flip (row, col) for (col, row) to match (width, height) image coordinates
  return np.flip(candidate_indices)

def plot_image_with_point(image_path, point, label):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(10, 10))
  plt.imshow(image)
  show_points(point, label, plt.gca())
  plt.savefig('first_frame_with_points.png')
  plt.axis('off')
#   plt.show() 

def first_menu(image_path, points, labels, mask, predictor):
    print('the first frame with selected points will be stored in first_frame_with_points.png and the corresponding mask to first_mask.png')
    plot_image_with_point(image_path, points, labels)
    cv2.imwrite('first_mask.png', mask * 255)
    print('done, you can check the image and mask and decide if I should label the rest.')
    response = ''
    while response not in {'improve', 'continue'}:
        response = input('improve mask or continue? (improve/continue): ')
    if response == 'improve':
        p1  = input('tell me the coordinates for point 1 e.g "x1 y1" or "same": ')
        p2 =  input('now tell me the coordinates for point 2 e.g "x2 y2" or "same": ')
        if p1 != 'same':
            x1, y1 = p1.split()
            points[0][0] = int(x1)
            points[0][1] = int(y1)
        if p2 != 'same':
            x2, y2 = p2.split()
            points[1][0] = int(x2)
            points[1][1] = int(y2)
           
        confirm = input(f"great, new points are: ({points[0][0]},{points[0][1]}) and ({points[1][0]}, {points[1][1]}), correct? (y/n): ")
        if confirm == 'n':
           exit

        print('generating new mask...')

        first_masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
        )
        best_first_mask = first_masks[np.argmax(scores), :, :]
        first_menu(image_path, points, labels, best_first_mask, predictor)
    else:
       return points
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This script written by sam uses SAM to label masks automatically""")
    parser.add_argument('-i','--input-folder', help='path to folder with images', required=True) #default="files_1_bad")
    parser.add_argument('-o','--output-folder', help='path to folder to store labels', required=True)# default="toremove.txt")
    parser.add_argument('-g','--gpu', help='use gpu (requires more than 4GB of gpu RAM)', default=False)# default="toremove.txt")

    args = vars(parser.parse_args())
    img_dir = args['input_folder']
    masks_dir = args['output_folder']
    gpu = True if args['gpu'] == 'True' else False
    print('loading SAM ...')
    vit_b_weights = "./weights/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=vit_b_weights)
    if gpu:
        device = "cuda"
        sam.to(device=device)

    predictor = SamPredictor(sam)

    os.makedirs(masks_dir, exist_ok=True)
    list_of_images = get_sorted_files(img_dir)
    p1_width, p1_height = 900, 1100
    p2_width, p2_height = 200, 700
    # p3_width, p3_height = 250, 750

    first_input_points = np.array([[p1_width, p1_height], [p2_width, p2_height]]) #, [p3_width, p3_height]])
    first_input_labels = np.ones(2)
    start_from = 0
    print('loading first frame and predicting its mask')
    first_image_path = os.path.join(img_dir, list_of_images[start_from])
    first_image = cv2.imread(first_image_path)
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB) 

    predictor.set_image(first_image)

    first_masks, scores, logits = predictor.predict(
    point_coords=first_input_points,
    point_labels=first_input_labels,
    multimask_output=True,
    )
    n_points = 2

    best_first_mask = first_masks[np.argmax(scores), :, :]
    input_points = get_road_indices_from_mask(best_first_mask, n_points)
    input_labels = np.ones(n_points)

    input_points = first_menu(first_image_path, first_input_points, first_input_labels, best_first_mask, predictor)

    for f in list_of_images[start_from:]:
        start = time.time()
        filename = os.path.join(img_dir, f)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        predictor.set_image(img)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )
        n_input_points = 50
        best_mask = masks[np.argmax(scores), :, :] 
        input_points = get_road_indices_from_mask(best_mask, n_points)
        input_labels = np.ones(n_points)

        file_n = get_number(f)
        mask_filename = f"{file_n}.png"
        cv2.imwrite(os.path.join(masks_dir, mask_filename), best_mask * 255)
        end = time.time()
        elapsed = end - start

        print(f"done with img {file_n} took {elapsed} seconds")