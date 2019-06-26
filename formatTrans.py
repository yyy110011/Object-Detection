import pandas as pd
import cv2
import os
from PIL import Image
import time
import numpy as np

class formatTrans():
    
    def __init__(self, data_addr, data_box_addr, class_descriptions_addr, output_addr):
        self.data_addr = data_addr
        self.data_box_addr = data_box_addr
        self.class_descriptions_addr = class_descriptions_addr
        self.output_addr = output_addr

    def run(self):

        output = open(self.output_addr, 'w')
        out_string = ''
        data = pd.read_csv(self.data_box_addr)
        
        # tag & index match
        tag2obj = pd.read_csv(self.class_descriptions_addr, names=['tag', 'object'])
        tag2obj.set_index('tag', inplace=True)
        tag2obj['i'] = list(range(len(tag2obj)))
        tag2obj = tag2obj.drop('object', axis=1)
        tag2obj = tag2obj.to_dict('index')
        for t in tag2obj:
            tag2obj[t] = tag2obj[t]['i']

        i = 0
        NAME = data["ImageID"].unique()

        for pic_name in NAME:
            pic_addr = self.data_addr + '/' +  pic_name + '.jpg'
            out_string += pic_addr + ' ' 
        # for each labeled box
            object_data = data.loc[data["ImageID"] == pic_name,:]
            object_data = np.array(object_data)
            #with Image.open(pic_addr) as pic:
            #        height, width = pic.size
            for j in range(len(object_data)):
                x_min = '%.2f' % (object_data[j][4])
                y_min = '%.2f' % (object_data[j][6])
                x_max = '%.2f' % (object_data[j][5])
                y_max = '%.2f' % (object_data[j][7])
                class_id = tag2obj[object_data[j][2]]
                box_string = x_min + ',' + y_min + ',' + x_max + ',' + y_max + ',' + str(class_id) + ' '
                
                out_string += box_string
            out_string += '\n'
            if i % 100 ==0:
                print(str(i) + ' of the file is finished')
            i += 1
            #if i == 100:
            #    break
        output.write(out_string)
        output.close()


if __name__ == "__main__":
    trans = formatTrans('/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/train_0',
                   'train-annotations-bbox.csv',
                   'class-descriptions-boxable.csv',
                   #'/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/transformed.txt')
                   'transformed.txt')
    trans.run()