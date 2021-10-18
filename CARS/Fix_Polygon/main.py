from PIL import Image
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib
#matrice de correction
path = "/Users/antoinerousseau/Desktop/background_CARS_20210913/corr_mat.csv"
#image à corriger
image_data = "/Users/antoinerousseau/Desktop/axon_tile_3x3/nul/Slice_6.tif"

def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        return file_array

image_raw = read_file(image_data)

corr_mat = genfromtxt(path, delimiter=',')

#contraste
#corr_mat[np.where(corr_mat > np.percentile(corr_mat, 50))] = 0
# corr_mat[:200:1, 0] = 0
# corr_mat[800::1, 0] = 0

N_row = 512


image = image_raw * corr_mat

#Remove problemes caused by polygon's scratched surfaces
def fix_polygon(image):
    for i in range(20, N_row, 36):
        image[i] = image[i+1]/2 + image[i-1]/2
        #corr[i] = np.percentile(corr, 0)

    for i in range(14, N_row, 36):
        image[i] = image[i+1]/2 + image[i-1]/2
        #corr[i] = np.percentile(corr, 0)
    return image

#image = fix_polygon(image)

# image = np.delete(image, range(800, 1024), 1)
# image = np.delete(image, range(0, 150), 1)


plt.imshow(image)
#plt.imshow(image)
#plt.imshow(image_raw, cmap='gray', vmin=50, vmax=250)
plt.show()

#save image as png
#matplotlib.image.imsave('kell_save_test2.png', image)
