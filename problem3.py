import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud, ImageColorGenerator


# read the pictures' name of positive and negative in lists
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


path_1 = 'CNN/positive/'
path_2 = 'CNN/negative/'

filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)

positive_list = []
negative_list = []

for filename in filelist_1:
    try:
        positive_list.append(filename[13:])
    except OSError as e:
        print(e.args)

for filename in filelist_2:
    try:
        negative_list.append(filename[13:])
    except OSError as e:
        print(e.args)

# open excel 2
df2 = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
df2_filename = df2['FileName']
df2_globalId = df2['GlobalID']

filename_list = np.array(df2_filename).tolist()
globalId_list = np.array(df2_globalId).tolist()

# get id of positive
positive_globalId_list = []
for index in positive_list:
    positive_globalId_list.append(globalId_list[filename_list.index(index)])

# get id of negative
negative_globalId_list = []
for index in negative_list:
    negative_globalId_list.append(globalId_list[filename_list.index(index)])

# open data set
df = pd.read_excel('2021MCMProblemC_DataSet.xlsx')

# find notes, latitude, longitude of the positive id
positive_note = []
positive_latitude = []
positive_longitude = []

for index in positive_globalId_list:
    positive_note += np.array(df.loc[(df['GlobalID'] == index), ['Notes']]).tolist()
    positive_latitude += np.array(df.loc[(df['GlobalID'] == index), ['Latitude']]).tolist()
    positive_longitude += np.array(df.loc[(df['GlobalID'] == index), ['Longitude']]).tolist()

positive_note_list = []
for index in positive_note:
    positive_note_list.append(index[0])

positive_latitude_list = []
for index in positive_latitude:
    positive_latitude_list.append(index[0])

positive_longitude_list = []
for index in positive_longitude:
    positive_longitude_list.append(index[0])

# find notes, latitude, longitude of the negative id
negative_note = []
negative_latitude = []
negative_longitude = []

for index in negative_globalId_list:
    negative_note += np.array(df.loc[(df['GlobalID'] == index), ['Notes']]).tolist()
    negative_latitude += np.array(df.loc[(df['GlobalID'] == index), ['Latitude']]).tolist()
    negative_longitude += np.array(df.loc[(df['GlobalID'] == index), ['Longitude']]).tolist()

negative_note_list = []
for index in negative_note:
    negative_note_list.append(index[0])

negative_latitude_list = []
for index in negative_latitude:
    negative_latitude_list.append(index[0])

negative_longitude_list = []
for index in negative_longitude:
    negative_longitude_list.append(index[0])

# draw word cloud
str_positive_note = ''
for index in positive_note_list:
    str_positive_note += index

cut_text = jieba.cut(str_positive_note)
result = "/".join(cut_text)
wc = WordCloud(background_color='white', width=800, relative_scaling=1,
               height=600, max_font_size=100,
               max_words=1000)
wc.generate(result)
plt.figure("positive notes word cloud")
plt.imshow(wc)
plt.axis("off")

str_negative_note = ''
for index in negative_note_list:
    str_negative_note += str(index)

cut_text = jieba.cut(str_negative_note)
result = "/".join(cut_text)
wc = WordCloud(background_color='white', width=800, relative_scaling=1,
               height=600, max_font_size=100,
               max_words=1000)
wc.generate(result)
plt.figure("negative notes word cloud")
plt.imshow(wc)
plt.axis("off")
plt.show()

