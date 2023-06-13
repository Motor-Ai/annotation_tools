# import os
# import shutil
# import argparse

# import xml.etree.ElementTree as ET

# def move_images_by_object(source_folders, destination_folder, objects_list):
#     """
#     :param source_folders: a list of folders, each folder has the two subfolders: images and labels
#     :param destination folder: a folder where the output folders will be saved
#     :param objects: a list of objects, for example: ["Pedestrian", "Stop", "Speed_Limit_30", "Speed_Limit_50"]
#     all the xml files will be sorted by the containing objects and the images will be saved in the corresponding folders 
#     """
#     # Create destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
    
#     # Get a list of all folders in the source directory
#     folders = source_folders
    
#     object_dict = {}
#     object_file_counter = {}
    
#     for obj in objects_list:
#         # Create destination folders for the object
#         object_folder = os.path.join(destination_folder, obj)
#         if os.path.exists(object_folder):
#             image_folder = os.path.join(object_folder, "images")
#             image_files = os.listdir(image_folder)  # Get the list of image files
#             max_image_number = max([int(file.split(".")[0]) for file in image_files])  # Extract the image number and find the maximum
#             object_file_counter[obj] = max_image_number
    
#         object_dict[obj] = object_folder
#         object_file_counter[obj] = 0
        
    
#     # Iterate over each folder
#     for folder in folders:
#         folder_path = folder
#         if os.path.isdir(folder_path):
#             # Iterate over each object
#                 # Get a list of all files in the images folder
#                 images_folder = os.path.join(folder_path, 'images')
#                 image_files = sorted(os.listdir(images_folder))
                
#                 # Iterate over each image file
#                 for image_file in image_files:
#                     image_path = os.path.join(images_folder, image_file)
                    
#                     # Get the corresponding XML file path
#                     xml_file = image_file.split('.')[0] + '.xml'
#                     xml_path = os.path.join(folder_path, 'labels', xml_file)
                    
#                        # Check if the XML file exists and contains any objects
#                     if os.path.isfile(xml_path):
#                         objects = get_objects(xml_path)
                        
#                         if objects:
#                             for obj in objects:
#                                 if obj in objects_list:
#                                     # Move the image file to the object folder
#                                     object_folder = object_dict[obj]
#                                     file_number = object_file_counter[obj]
#                                     if object_folder is not None:
#                                         object_image_folder = os.path.join(object_folder, 'images')
#                                         object_label_folder = os.path.join(object_folder, 'labels')
#                                         os.makedirs(object_image_folder, exist_ok=True)
#                                         os.makedirs(object_label_folder, exist_ok=True)
                                        
#                                         new_xml_name = f"{file_number}.xml"
#                                         new_xml_path = os.path.join(object_label_folder, new_xml_name)
                                        
#                                         new_image_name = f"{file_number}.png"
#                                         new_image_path = os.path.join(object_image_folder, new_image_name)
                                        
#                                         shutil.copy2(image_path, new_image_path)
#                                         shutil.copy2(xml_path, new_xml_path)
                                        
#                                         object_file_counter[obj] +=1
# def get_objects(xml_path):
#     # Parse the XML file
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
    
#     # Find all object elements
#     objects = root.findall('object')
    
#     # Create a list to store the objects
#     object_list = []
    
#     # Iterate over each object element
#     for obj_elem in objects:
#         # Get the name of the object
#         name_elem = obj_elem.find('name')
#         if name_elem is not None:
#             object_name = name_elem.text
            
#             # Add the object to the list
#             object_list.append(object_name)
    
#     return object_list


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source_folders', nargs='+', type=str, default = ['dataset/sc4'], help='List of source folders')
#     parser.add_argument('--objects_list', nargs='+', type=str, default = ["Pedestrian", "Stop", "Speed_Limit_30", "Speed_Limit_50"], help='List of objects for which to crate the folders')
#     parser.add_argument('--destination_folder', type=str, default = 'sorted_dataset', help='List of source folders')
#     args = parser.parse_args()
# # Call the function to move images by object
#     move_images_by_object(args.source_folders, args.destination_folder, args.objects_list)
    
import os
import xml.etree.ElementTree as ET

data_folder = 'sorted_dataset'

# Iterate over the folders and subfolders
for dirs in os.listdir(data_folder):
        folder_path = os.path.join('sorted_dataset', dirs)
        xml_folder = os.path.join(folder_path, 'labels')

        # Iterate over the XML files in the labels folder
        for xml_file in os.listdir(xml_folder):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root_et = tree.getroot()

            # Find the width and height elements and switch their values
            width = root_et.find('size/width')
            height = root_et.find('size/height')
            width.text, height.text = str(2592), str(2048)

            # Save the modified XML file
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

print("Width and height switched for all XML files.")