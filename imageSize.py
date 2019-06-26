from PIL import Image
import pandas as pd
import os


data_addr = '/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/train_0'
data_box_addr = 'train-annotations-bbox.csv'
class_descriptions_addr = 'class-descriptions-boxable.csv'
output_addr = '/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/transformed.txt'



output = open(output_addr, 'w')
out_string = ''
data = pd.read_csv(data_box_addr)
tag2obj = pd.read_csv(class_descriptions_addr, names=['tag', 'object'])
i = 0
NAME = data["ImageID"].unique()
H = []
W = []

i = 0
for pic_name in NAME:
    pic_addr = os.path.join(data_addr, pic_name + '.jpg')
    out_string += pic_addr + ' ' 
# for each labeled box
    object_data = data.loc[data["ImageID"] == pic_name,:]
    with Image.open(pic_addr) as pic:
        height, width = pic.size
    H.append(height)
    W.append(width)
    i = i + 1
    if i % 100 == 0:
        print(str(i) + ' of image is read!')

NAME = pd.DataFrame(NAME, columns=["ImageID"])
NAME["height"] = pd.DataFrame(H)
NAME["width"] = pd.DataFrame(W)

NAME.to_csv("train_0_size.csv")