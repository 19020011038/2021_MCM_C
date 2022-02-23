import shutil

import pandas as pd
import numpy as np
import os
from PIL import Image


# read the file
df = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
df2 = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')

# collect the id of report
globalIds_positive = df.loc[(df['Lab Status'] == 'Positive ID'), ['GlobalID']]
globalIds_negative = df.loc[(df['Lab Status'] == 'Negative ID'), ['GlobalID']]
globalIds_unverified = df.loc[(df['Lab Status'] == 'Unverified'), ['GlobalID']]

ndata = np.array(globalIds_positive)
globalIdList_positive = ndata.tolist()
ndata = np.array(globalIds_negative)
globalIdList_negative = ndata.tolist()
ndata = np.array(globalIds_unverified)
globalIdList_unverified = ndata.tolist()


# find the name of the pictures according to id which are positive
fileNameList_all_temp_positive = []
for index in globalIdList_positive:
    id = index[0]
    fileName = df2.loc[(df2['GlobalID'] == id) & (df2['FileType'] == 'image/jpg'), ['FileName']]
    ndata = np.array(fileName)
    fileNameList_positive = ndata.tolist()
    fileNameList_all_temp_positive += fileNameList_positive

fileNameList_all_positive = []
for index in fileNameList_all_temp_positive:
    fileNameList_all_positive.append(index[0])

# find the name of the pictures according to id which are negative
fileNameList_all_temp_negative = []
for index in globalIdList_negative:
    id = index[0]
    fileName = df2.loc[(df2['GlobalID'] == id) & (df2['FileType'] == 'image/jpg'), ['FileName']]
    ndata = np.array(fileName)
    fileNameList_negative = ndata.tolist()
    fileNameList_all_temp_negative += fileNameList_negative

fileNameList_all_negative = []
for index in fileNameList_all_temp_negative:
    fileNameList_all_negative.append(index[0])

# find the name of the pictures according to id which unverified
fileNameList_all_temp_unverified = []
for index in globalIdList_unverified:
    id = index[0]
    fileName = df2.loc[(df2['GlobalID'] == id) & (df2['FileType'] == 'image/jpg'), ['FileName']]
    ndata = np.array(fileName)
    fileNameList_unverified = ndata.tolist()
    fileNameList_all_temp_unverified += fileNameList_unverified

fileNameList_all_unverified = []
for index in fileNameList_all_temp_unverified:
    fileNameList_all_unverified.append(index[0])


# read the pictures
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


path = '2021MCM_ProblemC_Files/'
filelist = read_image(path)

num_positive = 0
num_negative = 0

# save the pictures in sets
shutil.rmtree('CNN/positive_copy')
os.mkdir('CNN/positive_copy')
# shutil.rmtree('CNN/positive')
# os.mkdir('CNN/positive')

for filename in filelist:
    try:
        im = Image.open(filename)
        ss = filename[23:]
        if ss in fileNameList_all_positive:
            im.save('CNN/positive/' + ss)
            num_positive += 1
        if ss in fileNameList_all_negative:
            im.save('CNN/negative/' + ss)
            im.save('CNN/negative_copy/' + ss)
            num_negative += 1
        if ss in fileNameList_all_unverified:
            im.save('CNN/unverified/' + ss)
    except OSError as e:
        print(e.args)

print(num_positive)
print(num_negative)
range_num = num_negative / num_positive
ran = int(range_num)

# copy positive pictures as training set
path_2 = 'CNN/positive/'
filelist_2 = read_image(path_2)

for i in range(ran):
    for filename in filelist_2:
        try:
            im = Image.open(filename)
            ss = filename[13:]
            im.save('CNN/positive_copy/' + ss[0:-4] + '_' + str(i) + '.jpg')
        except OSError as e:
            print(e.args)
