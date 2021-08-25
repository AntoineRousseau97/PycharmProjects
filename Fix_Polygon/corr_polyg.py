from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#path de l'image à corriger
image_raw = "/Volumes/Goliath/labdata/arousseau/CARS/experimental data/2021.05.18/kelly1.18.05.2021-001-Bliq VMS-ch-2-Time_0.008s-Frame_008800.tif"

#lecture du fichier .tiff pour obtenir l'image sous forme de array
def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        return file_array

image = read_file(image_raw)

#Remove problemes caused by polygon's scratched surfaces
def fix_polygon(image):
    for i in range(34, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2
        #corr[i] = np.percentile(corr, 0)

    for i in range(6, 512, 36):
        image[i] = image[i+1]/2 + image[i-1]/2
        #corr[i] = np.percentile(corr, 0)
    return image

clean_image = fix_polygon(image)
plt.imshow(clean_image)
plt.show()