from PIL import Image, ImageDraw
from main import model, PATH, device
import torch
import numpy as np
import matplotlib
import torch.nn.functional as nnf

"""
water = Image.open("..\\dataset\\water\\1_Волоколамск_13.jpeg").convert("RGBA")
roads = Image.open("..\\dataset\\roads\\2_Волоколамск_13.jpeg").convert("RGBA")
roads_federal = Image.open("..\\dataset\\roads\\5_Волоколамск_13.jpeg").convert("RGBA")
railway = Image.open("..\\dataset\\railway\\3_Волоколамск_13.jpeg").convert("RGBA")
city = Image.open("..\\dataset\\city\\4_Волоколамск_13.jpeg").convert("RGBA")
"""
#"""
water = Image.open("..\\dataset\\test\\water\\1_Электроугли_23.jpeg").convert("RGBA")
roads = Image.open("..\\dataset\\test\\roads\\2_Электроугли_23.jpeg").convert("RGBA")
roads_federal = Image.open("..\\dataset\\test\\roads\\5_Электроугли_23.jpeg").convert("RGBA")
railway = Image.open("..\\dataset\\test\\railway\\3_Электроугли_23.jpeg").convert("RGBA")
city = Image.open("..\\dataset\\test\\city\\4_Электроугли_23.jpeg").convert("RGBA")


total = city.copy()
for img in [roads,roads_federal, railway, water]:
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] > 230 and item[1] > 230 and item[2] > 230:
            newData.append((item[0], item[1], item[2], 0))
        else:
            newData.append(item)
    img.putdata(newData)
    Image.Image.paste(total, img, (0, 0), img)

total.save("total.png", "PNG")
#total.save("real.png", "PNG")
#"""

chunk_cnt_per_side = 110
chunk_size = 64
total = Image.open("total.png").convert("RGB")
total2 = Image.open("total.png").convert("RGB")
model.load_state_dict(torch.load(PATH))
model.to(device)
dataset = torch.Tensor(np.load("..\\dataset\\test_data_64.npy"))
dataset = dataset[chunk_cnt_per_side**2 * 2:chunk_cnt_per_side**2 * 3]
#dataset = dataset[:chunk_cnt_per_side**2]
dataset = dataset.to(device)
draw = ImageDraw.Draw(total, "RGBA")
"""
real_total = Image.open("real.png").convert("RGB")
real_answers = torch.Tensor(np.load("..\\dataset\\targets_64.npy"))
real_answers = real_answers[chunk_cnt_per_side**2:chunk_cnt_per_side**2 * 2]
real_draw = ImageDraw.Draw(real_total, "RGBA")
#"""
draw2 = ImageDraw.Draw(total2, "RGBA")
norm = matplotlib.colors.PowerNorm(gamma=4, vmin=0, vmax=1)
cmap = matplotlib.colormaps["plasma"]#winter
#print(norm(0.5), norm(0.9), norm(1), norm(0.001))
for i in range(chunk_cnt_per_side):
    for j in range(chunk_cnt_per_side):
        result = nnf.softmax(model(torch.unsqueeze(dataset[i * chunk_cnt_per_side + j], 0)), dim=1)
        answer = cmap(norm(result[0][1].item()))
        color = tuple([int(answer[i]*255) for i in range(3)] + [255])#127
        draw.rectangle((i *chunk_size, j* chunk_size, (i + 1) * chunk_size, (j + 1) * chunk_size), fill=color)
        color2 = tuple([int(answer[i] * 255) for i in range(3)] + [127])  # 127
        draw2.rectangle((i *chunk_size, j* chunk_size, (i + 1) * chunk_size, (j + 1) * chunk_size), fill=color2)
        #real_color = (57,255,20,127) if real_answers[i * chunk_cnt_per_side + j] == 1 else (0,0,0,0)
        #real_draw.rectangle((i * chunk_size, j * chunk_size, (i + 1) * chunk_size, (j + 1) * chunk_size), fill=real_color)

total.save("result_coal.png", "PNG")
total2.save("coal.png", "PNG")
#real_total.save("real.png", "PNG")
total.show()



