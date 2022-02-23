import pandas as pd
import numpy as np
import os
from PIL import Image


# read the pictures
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


path = 'CNN/positive/'
filelist = read_image(path)

for i in range(232):
    for filename in filelist:
        try:
            im = Image.open(filename)
            ss = filename[23:]
            im.save('CNN/temp/' + ss[0:-4] + '_' + str(i) + '.jpg')
        except OSError as e:
            print(e.args)

path = 'CNN/temp/'
filelist = read_image(path)
for i in range(2):
    for filename in filelist:
        try:
            im = Image.open(filename)
            im.save('CNN/positive_copy/' + filename[9:-4] + '_' + str(i) + '.jpg')
        except OSError as e:
            print(e.args)


path = 'CNN/negative/'
filelist = read_image(path)
for i in range(2):
    for filename in filelist:
        try:
            im = Image.open(filename)
            im.save('CNN/negative_copy/' + filename[13:-4] + '_' + str(i) + '.jpg')
        except OSError as e:
            print(e.args)
