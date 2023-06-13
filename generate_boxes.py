import os
import argparse
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert

import xml.etree.ElementTree as ET


def write_bbox_to_xml(bbox_annotation, labels, logits, output_file, filename, source, img_dims, box_threshold, text_threshold, text_prompt,
                      cleaned=0, bbox_format = 'cxcywh'):
    # Create the XML structure
    # header
    annotation = ET.Element("annotation")
    fn = ET.SubElement(annotation, "filename")
    fn.text = filename
    src = ET.SubElement(annotation, "source")
    src.text = source
    size = ET.SubElement(annotation, "size")
    width_ =  ET.SubElement(size, "width")
    width_.text = str(img_dims[1])
    height_ =  ET.SubElement(size, "height")
    height_.text = str(img_dims[0])
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

        if bbox_format=='cxcywh':
            xyxy_bbox =  box_convert(boxes=bbox_annotation[i], in_fmt="cxcywh", out_fmt="xyxy")
        else:
            xyxy_bbox = bbox_annotation[i]
            
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmax = ET.SubElement(bndbox, "xmax")
        xmin.text = str(float(xyxy_bbox[0]))
        xmax.text = str(float(xyxy_bbox[2]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymax = ET.SubElement(bndbox, "ymax")
        ymin.text = str(float(xyxy_bbox[1]))
        ymax.text = str(float(xyxy_bbox[3]))
        
        area = bbox_annotation[i][2].item() * bbox_annotation[i][3].item() * img_dims[0] * img_dims[1]
        bbox_area = ET.SubElement(obj, "bbox_area")
        bbox_area.text = str(area)
        

    # Write the XML to a file
    tree = ET.ElementTree(annotation)
    with open(output_file, "wb") as f:
        f.write(b'<?xml version="1.0" ?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-folder', help='input folder with images')
    parser.add_argument('-o','--output-folder', help='output folder where labels are stored')
    parser.add_argument('-p','--prompt', help='text prompt specifying objects to detect', default="traffic sign .")
    parser.add_argument('-t','--text-threshold', help='threshold for the text prompt', default="0.27")
    parser.add_argument('-b','--box-threshold', help='threshold for the box ', default="0.29")


    args = vars(parser.parse_args())

    image_path = args['input_folder']
    output_path = args['output_folder']
    TEXT_PROMPT = args['prompt']
    TEXT_TRESHOLD = float(args['text_threshold'])
    BOX_TRESHOLD = float(args['box_threshold'])

    os.makedirs(output_path, exist_ok=True)
    weights_path = "weights/groundingdino_swint_ogc.pth"
    model_path = "models/GroundingDINO_SwinT_OGC.py"
    model = load_model(model_path, weights_path)
    # img_dims = 1900, 1200, 3

    for file in tqdm(os.listdir(image_path)):
    # for file in os.listdir(image_path):
        root_path, image_folder = os.path.split(image_path)
        img_filename = os.path.join(image_path, file)
        output_img_filename = os.path.join(output_path, file)
        label_filename = file.replace('.png', '.xml')
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
                          img_dims=img_dims, box_threshold=BOX_TRESHOLD, 
                          text_threshold=TEXT_TRESHOLD,
                          text_prompt=TEXT_PROMPT)
        
        # boxes, phrases, logits = get_annotation_from_xml(label_path)
        # cv2.imwrite(output_img_filename.replace('.jpg', '_label.jpg'), annotated_frame)

