import os
import random
import shutil

# Define the path to the main folder
main_folder = "dataset/sc3"
images_folder = os.path.join(main_folder, "images")
labels_folder = os.path.join(main_folder, "labels")

# Define the path to the new folder where the images without objects will be copie
new_folder = "sorted_dataset/empty"
os.makedirs(new_folder, exist_ok=True)

new_images_folder = "sorted_dataset/empty/images"
new_labels_folder = "sorted_dataset/empty/labels"
os.makedirs(new_images_folder, exist_ok=True)
os.makedirs(new_labels_folder, exist_ok=True)


image_files = os.listdir(images_folder)
random.shuffle(image_files)
count = 0

for image_file in image_files:
    # Construct the path to the corresponding label file
    label_file = os.path.join(labels_folder, os.path.splitext(image_file)[0] + ".xml")

    # Read the label file
    with open(label_file, "r") as file:
        content = file.read()

    # Check if the label file contains any objects
    if "<object>" not in content:
        count += 1

        # Construct the new image and label names with a consistent naming scheme
        new_image_name = f"{count}.png"
        new_label_name = f"{count}.xml"

        # Construct the paths to the source image file and the destination file in the new image folder
        source_image_path = os.path.join(images_folder, image_file)
        destination_image_path = os.path.join(new_images_folder, new_image_name)

        # Copy the image file to the new image folder with the new name
        shutil.copyfile(source_image_path, destination_image_path)

        # Construct the paths to the source label file and the destination file in the new labels folder
        source_label_path = label_file
        destination_label_path = os.path.join(new_labels_folder, new_label_name)

        # Copy the label file to the new labels folder with the new name
        shutil.copyfile(source_label_path, destination_label_path)

        # Update the filename inside the copied label file
        with open(destination_label_path, "r+") as file:
            content = file.read()
            file.seek(0)
            file.write(content.replace(os.path.basename(image_file), new_image_name))
            file.truncate()

        # Break the loop if we have reached the desired number of images without objects
        if count == 150:
            break
