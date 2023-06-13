# Annotation Tools
This project contains:
1. Scripts that use big pre-trained neural networks to create annotations. 
2. Scripts that allow the user to visualize and clean the labels generated automatically.

**Installation:**
```bash
git clone git@github.com:Motor-Ai/annotation_tools.git
```
## 2D Bounding Boxes
### Models
- [GroundingDINO] (https://github.com/IDEA-Research/GroundingDINO)

**Installation:**

Install GroundingDINO in your desired conda environment and download the model weigths.

```bash
conda env create -f sam.yml

conda activate sam
cd <path to where you store your repositories>
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip3 install -q -e .
cd <path to annotation_tools>
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Labeling Data
```
python generate_boxes.py -i <path to images> -o <path to folder to store labels>  -p <text prompt> -b <box threshold> -t <text threshold>
```
### Cleaning Labels
```
# This command assumes that the data is stored in the following way:
<!--  path_to_your_data
 - labels
 - images -->


python clean_boxes.py -images_folder <path to your data>
```
### The functionality 0f the app:

| Control  | Command  | 
|---|---|
| Right Arrow  | load next image  |  
| Left Arrow | load previous image  |    
|  Left Click | delete bounding box  |    
|  Right Click (Hold and Release) | create bounding box  |       
|  Mouth Click (Hold and Release) | zoom into a region  |  
| a  | start autoplay of images  |  
| s | stop autoplay of images  |    
| d | delete all bounding boxes from the image  |    
| q | next image and delete all bounding boxes from the previous one |   


## Segmentation Masks
### Models
- [SAM] (https://github.com/facebookresearch/segment-anything)

**Installation:**
```
conda activate <your conda environment>
pip install git+https://github.com/facebookresearch/segment-anything.git
cd <path to annotation_tools>/weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Labeling Data
To start labeling run the following script and follow the instructions thereafter:
```
python generate_masks.py -i <path to images folder> -o <path to folder to store masks> 
```

### Cleaning Data
For now you need to move your images and corresponding masks to the datastructure mentioned above before visualizing the data.
To clean the generated masks run:

```
python clean_masks.py -s <session name> -t <train or val> --camera-index <0 or 1>
```
You can see the controls on the bottom bar of the app.
