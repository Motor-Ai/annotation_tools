# Annotation Tools
**Installation:**
```bash
git clone git@github.com:Motor-Ai/annotation_tools.git
```
## Datasets structure
```
├── train
│   ├── <session id>
│   │   ├── images
│   │   └── labels
│    ...
│   ├── <session id>
│       ├── images
│       └── labels
└── val
    └── <session id
        ├── images
        └── labels
```

## 2D Bounding Boxes
### Models
- [GroundingDINO] (https://github.com/IDEA-Research/GroundingDINO)

**Installation:**

Install GroundingDINO in your desired conda environment and download the model weigths.

```bash
conda activate <your conda environment>
cd <path to where you store your repositories>
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip3 install -q -e .
cd <path to annotation_tools>
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Labelling Data
```
python generate_boxes.py -i <path to images> -o <path to folder to store labels>  -p <text prompt specifying models to detect e.g 'traffic sign .'>
```
### Cleaning Labels
```
# This command assumes that the data is stored according to the datasets structure shown above.
python clean_boxes.py -s <session id> -t <data type (train or val)>
```

## Segmentation Masks
### Models
- [SAM] (https://github.com/facebookresearch/segment-anything)

