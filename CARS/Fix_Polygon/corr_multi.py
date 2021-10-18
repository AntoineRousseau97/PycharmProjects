from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import cv2

# TODO path du folder contenant les images à corriger
image_raw_path = "/Users/antoinerousseau/Desktop/test_stitch1"
# TODO path du folder where to save les images corrected
path = "/Users/antoinerousseau/Desktop/new_vidange/"

# fetch files name
def listNameOfFiles(directory: str, extension="tif") -> list:
    foundFiles = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, f'*.{extension}'):
            foundFiles.append(file)
    return foundFiles

files = listNameOfFiles(image_raw_path)

# lecture du fichier .tiff pour obtenir l'image sous forme de array
def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        return file_array

# Remove black lines caused by polygon's scratched surfaces
def fix_polygon(image):
    for i in range(30, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    for i in range(22, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    return image

x = []
for file in os.listdir(image_raw_path):
    # Check whether file is in tif format or not
    if file.endswith(".tif"):
        file_path = f"{image_raw_path}/{file}"
        # call read text file function
        x.append(read_file(file_path))

def change_im_size(image, low_cut=40, high_cut=552):
    new_image = []
    for i in image:
        i = i[low_cut:high_cut]
        new_image.append(i)
    return new_image

images = []
for i in x:
    images.append(change_im_size(fix_polygon(i)))

# fixed_images = []
# for i in images:
#     fixed_images.append(change_im_size(i))

print(files)

# save corrected images
for ind, im in enumerate(images):
    #plt.imsave(path+f'{files[ind]}', im, format='', cmap='gray')
    cv2.imwrite(path+f'{files[ind]}', np.array(im))
