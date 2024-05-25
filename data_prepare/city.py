import skimage as ski
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

for filename in glob.glob("..\\dataset\\city\\*.jpeg"):
    if filename.find("_p.jpeg") > 0:
        continue
    if os.path.isfile(filename.replace(".jpeg", "_p.jpeg")):
        print(filename.replace(".jpeg", "_p.jpeg") + " already exist")
        continue
    im = Image.open(filename)
    im2 = ski.color.rgb2gray(im)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(im)
    #axs[1].imshow(im2, cmap=plt.cm.gray)
    thresh = ski.filters.threshold_mean(np.array(im2))
    im3 = im2<=thresh
    axs[1].imshow(im3, cmap=plt.cm.gray)
    im4 = ski.morphology.closing(im3)
    im5 = ski.morphology.area_closing(im4, connectivity=1)
    axs[2].imshow(im5, cmap=plt.cm.gray)
    img = Image.fromarray(np.uint8(im5 * 255), mode="L")
    img.save(filename.replace(".jpeg", "_p.jpeg"), 'JPEG')
    print(filename.replace(".jpeg", "_p.jpeg") + " done")
#plt.show()