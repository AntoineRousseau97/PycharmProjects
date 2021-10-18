from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch
from numpy import genfromtxt
import cv2


# TODO path du folder contenant les images à corriger
image_raw_path = "/Users/antoinerousseau/Desktop/test_stitch1/"
# TODO path du folder where to save les images corrected
path = "/Users/antoinerousseau/Desktop/stitching/new_vidange/raw/"
# TODO path du folder where le corr_mat est
path_corr_mat = "/Users/antoinerousseau/Desktop/stitching/background_CARS_20210913/corr_mat.csv"

corr_mat = genfromtxt(path_corr_mat, delimiter=',')

# fetch files name
def listNameOfFiles(directory: str, extension="tif") -> list:
    foundFiles = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, f'*.{extension}'):
            foundFiles.append(file)
    return foundFiles

dirs = os.listdir(image_raw_path)
# for file in dirs:
#    print(file)

files = listNameOfFiles(image_raw_path)

# lecture du fichier .tiff pour obtenir l'image sous forme de array
def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        return file_array

def fix_polygon(image):
    for i in range(36, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    for i in range(8, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2

    return image

def change_im_size(image, low_cut=40):
    high_cut = low_cut+512
    new_image = []
    for i in image:
        i = i[low_cut:high_cut]
        new_image.append(i)
    return new_image

x = []
for file in os.listdir(image_raw_path):
    # Check whether file is in tif format or not
    if file.endswith(".tif"):
        file_path = f"{image_raw_path}/{file}"
        # call read text file function
        x.append(read_file(file_path))

images = []
for i in x:
    images.append(change_im_size(fix_polygon(np.int_(i))))

split = []
for i in files:
    split.append(i.split('-'))
# print(split)

# for i in files:
#     os.rename(image_raw_path + i, image_raw_path + i.split('-')[1] + '.tif')

# save corrected images
for ind, im in enumerate(images):
    #plt.imsave(path+f'{files[ind]}', im, format='', cmap='gray')
    cv2.imwrite(path+f'{files[ind].split("-")[1]}.tif', np.array(im))
    print(files[ind].split("-")[1])
    print(files[ind])