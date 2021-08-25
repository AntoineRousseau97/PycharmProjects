from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch

# TODO path du folder contenant les images à corriger
image_raw_path = "/Volumes/Goliath/labdata/arousseau/CARS/experimental data/2021.05.18/"
# TODO path du folder where to save les images corrected
path = "/Users/antoinerousseau/Desktop/vidange/"

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
    for i in range(34, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    for i in range(6, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    return image

x = []
for file in os.listdir(image_raw_path):
    # Check whether file is in tif format or not
    if file.endswith(".tif"):
        file_path = f"{image_raw_path}/{file}"
        # call read text file function
        x.append(read_file(file_path))

images = []
for i in x:
    images.append(fix_polygon(i))

# save corrected images
for ind, im in enumerate(images):
    plt.imsave(path+f'{files[ind].replace(".tif",".png")}', im)