import os
import argparse
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate

from utils import write_bbox_to_xml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-folder', help='input folder with images')
    parser.add_argument('-o','--output-folder', help='output folder where labels are stored')
    parser.add_argument('-p','--prompt', help='text prompt specifying objects to detect', default="traffic sign .")
    parser.add_argument('-t','--text-threshold', help='threshold for the text prompt', default="0.25")
    parser.add_argument('-b','--box-threshold', help='threshold for the box ', default="0.26")


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
                          img_dims=img_dims, box_threshold=BOX_TRESHOLD, 
                          text_threshold=TEXT_TRESHOLD,
                          text_prompt=TEXT_PROMPT)
        
        # boxes, phrases, logits = get_annotation_from_xml(label_path)
        # cv2.imwrite(output_img_filename.replace('.jpg', '_label.jpg'), annotated_frame)

