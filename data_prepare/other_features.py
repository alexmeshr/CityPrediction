import copy
import skimage as ski
#from skimage import color
#from skimage import filters
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numba import jit, prange, njit
from numba import cuda
import glob
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 697486795

@jit(nopython=True, target_backend='cuda')
def fill_im(im3, im4, ilen, jlen, radius):
    max_d = np.sqrt((radius)**2 +(radius)**2)
    for i, j in np.ndindex(im3.shape):
        if not im3[i, j]:
                for di, dj in np.ndindex(radius, radius):
                    d = 1 - np.sqrt(np.sqrt((di)**2 +(dj)**2)/max_d)
                    for pair in [(i + di, j + dj), (i - di, j + dj), (i + di, j - dj), (i - di, j - dj)]:
                        if pair[0] < ilen and pair[1] < jlen and pair[0] > 0 and pair[1] > 0:
                            im4[pair] = d if im4[pair] < d else im4[pair]
    return im4

@jit(nopython=True, parallel=True)#'float64[:,:](bool_[:,:], float64[:,:], int64, int64, int64)',
def fill_im_parallel(im3, im4, ilen, jlen, radius):
    max_d = np.sqrt((radius)**2 +(radius)**2)
    for ij in range(im3.shape[0]*im3.shape[1]):
        i = ij // im3.shape[1]
        j = ij % im3.shape[1]
        if not im3[i, j]:
                for k in prange(radius*radius):
                    di = k // radius
                    dj = k % radius
                    d = 1 - np.sqrt(np.sqrt((di)**2 +(dj)**2)/max_d)
                    for pair in [(i + di, j + dj), (i - di, j + dj), (i + di, j - dj), (i - di, j - dj)]:
                        if pair[0] < ilen and pair[1] < jlen and pair[0] > 0 and pair[1] > 0:
                            im4[pair] = d if im4[pair] < d else im4[pair]
    return im4

features = {"railway\\3*[!_p].jpeg":400, "water\\1*[!_p].jpeg": 300, "roads\\2_*[!_p].jpeg":100, "roads\\5_*[!_p].jpeg":200,"city\\4_*[!_p].jpeg" : 150}
for path in features:
    for filename in glob.glob(os.path.join("..\\dataset",  path)):
        if os.path.isfile(filename.replace(".jpeg", "_p.jpeg")):
            print(filename.replace(".jpeg", "_p.jpeg") + " already exist")
            continue
        print(filename + " start")
        im = Image.open(filename)
        im2 = ski.color.rgb2gray(im)
        im2[:,0] = 1
        im2[0,:] = 1
        im2[len(im2)-1,:] = 1
        im2[len(im2)-2,:] = 1
        #im2 = im.quantize(colors=2 )
        #print(im2.getcolors())
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(im)
        #axs[1].imshow(im2, cmap=plt.cm.gray)
        thresh = ski.filters.threshold_mean(np.array(im2))
        im3 = im2>=thresh
        notim3 = im2<=thresh
        axs[1].imshow(notim3, cmap=plt.cm.gray)
        im4 = np.zeros(im3.shape)
        #im4 = fill_im(im3, im4, im3.shape[0], im3.shape[1], features[path])
        im4 = fill_im_parallel(im3, im4, im3.shape[0], im3.shape[1], features[path])
        axs[2].imshow(im4, cmap=plt.cm.gray, vmin=0, vmax=1)
        img = Image.fromarray(np.uint8(im4 * 255), mode="L")
        img.save(filename.replace(".jpeg", "_p.jpeg"), 'JPEG')
        print(filename.replace(".jpeg", "_p.jpeg") + " done")
