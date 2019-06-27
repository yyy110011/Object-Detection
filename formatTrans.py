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
        NAME = np.array(os.listdir(self.data_addr))#data["ImageID"].unique()
        data = data.groupby('ImageID')
        T1 = time.time()
        for pic_name in NAME:
            pic_ID = pic_name[:-4]
            #t1 = time.time()
            pic_addr = self.data_addr + '/' +  pic_name
            out_string += pic_addr + ' ' 
        # for each labeled box
            object_data = data.get_group(pic_ID)
            #t1_1 = time.time()
            object_data = np.array(object_data)
            #t2 = time.time()
            with Image.open(pic_addr) as pic:
                    height, width = pic.size
            for j in range(len(object_data)):
                x_min = int(object_data[j][4] * width)
                y_min = int(object_data[j][6] * height)
                x_max = int(object_data[j][5] * width)
                y_max = int(object_data[j][7] * height)
                if x_max - x_min <= 0:
                    x_max = x_min + 1
                if y_max - y_min <= 0:
                    y_max = y_min + 1

                class_id = tag2obj[object_data[j][2]]
                box_string = str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(class_id) + ' '
                
                out_string += box_string
            out_string += '\n'
            output.write(out_string)
            out_string = ''
            #t3 = time.time()
            #print('after loc :'+ str(t1_1 - t1) + 'after np array:' + str(t2 - t1_1) + '  after box loop:' + str(t3 - t2))

            if i % 10000 ==0:
                print(str(i) + ' of the file is finished')
            i += 1
            #if i == 100000:
            #    break
        
        T2 = time.time()
        print("Time cose: " + str(T2 - T1))
        output.close()

    def out_class(self):
        out = ''
        out_addr_ptr = open('./class.txt', 'w')
        tag2obj = pd.read_csv(self.class_descriptions_addr, names=['tag', 'object'])
        for i in tag2obj['object']:
            out += i + '\n'
        out_addr_ptr.write(out)



if __name__ == "__main__":
    trans = formatTrans('/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/train_0',
                   'train-annotations-bbox.csv',
                   'class-descriptions-boxable.csv',
                   #'/media/yyy/D4EC13B5EC13913A/DLProj/KaggleCompetition/transformed.txt')
                   'transformed.txt')
    if 1:
        trans.run()
    if 0:
        trans.out_class()
