import os
import glob
import re
import numpy as np
from PIL import Image
#from convert_pdf import chunk_size, chunk_cnt_per_side
chunk_size = 64
chunk_cnt_per_side = 1
treshold = 6
for file in glob.glob("test/*/*_p.jpeg", recursive=True):
    img = Image.open(file)
    chunk_cnt_per_side = img.size[0]//chunk_size
    print(chunk_cnt_per_side)
    break
#chunks_dir = "chunks"
#if not os.path.exists(chunks_dir):
#    os.makedirs(chunks_dir)

city_cnt = 0
feature_cnt = 5
city_feature_index = 4

for file in glob.glob("test/*/*_p.jpeg", recursive=True):
    #print(file)
    city_cnt+=1
city_cnt //= feature_cnt
print(f"cities: {city_cnt}")
dataset = np.zeros((chunk_cnt_per_side**2 * city_cnt, feature_cnt, chunk_size, chunk_size))
old_city = np.zeros((chunk_cnt_per_side**2 * city_cnt,))
new_city = np.zeros((chunk_cnt_per_side**2 * city_cnt,))

feature_iters = np.zeros(feature_cnt+1, np.int_)

for file in glob.glob("test/*/*_p.jpeg", recursive=True):
    print(file)
    file_num = int(re.split('\\\|_', file)[2])
    img = Image.open(file)
    for i in range(chunk_cnt_per_side):
        for j in range(chunk_cnt_per_side):
            box = ( i *chunk_size, j* chunk_size, (i + 1) * chunk_size, (j + 1) * chunk_size)
            crop = img.crop(box).convert('L')
            arr = np.array(crop).astype(np.float32)/255.0
            if file_num <= feature_cnt:
                #print(feature_iters[file_num-1]*chunk_cnt_per_side**2 + (i * chunk_cnt_per_side + j))
                dataset[feature_iters[file_num-1]*chunk_cnt_per_side**2 + (i * chunk_cnt_per_side + j), file_num-1] = arr
                if file_num == city_feature_index:
                    old_city[feature_iters[file_num-1]*chunk_cnt_per_side**2 + (i * chunk_cnt_per_side + j)] = np.count_nonzero(arr==1)
            else:
                new_city[feature_iters[file_num-1]*chunk_cnt_per_side**2 + (i * chunk_cnt_per_side + j)] = np.count_nonzero(arr==1)
            #crop.save(chunks_dir + "\\" + str(file_num) + "_" + str(i * chunk_cnt_per_side + j) + ".png", 'PNG')
    feature_iters[file_num-1] += 1
    
targets = (new_city>old_city+treshold).astype(np.int32)
print(targets.shape, 100.0*np.count_nonzero(targets)/targets.shape[0],'%')
np.save("test_data_"+str(chunk_size), dataset)
#np.save("test_targets_"+str(chunk_size), targets)
