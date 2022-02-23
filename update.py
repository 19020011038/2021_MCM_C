import os
from PIL import Image
import shutil


def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


path_1 = 'CNN/positive_copy/'
path_2 = 'CNN/negative_copy/'
filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)
len_positive = len(filelist_1)
len_negative = len(filelist_2)
ran_f = len_negative / len_positive
ran = int(ran_f)

path_3 = 'CNN/positive/'
filelist_3 = read_image(path_3)
path_4 = 'CNN/negative/'
filelist_4 = read_image(path_4)
len_positive_o = len(filelist_3)
len_negative_o = len(filelist_4)
ran_f_o = len_negative_o / len_positive_o
ran_o = int(ran_f_o)


if ran_f > 4.0:

    # update training set
    shutil.rmtree('CNN/positive_copy')
    os.mkdir('CNN/positive_copy')
    shutil.rmtree('CNN/negative_copy')
    os.mkdir('CNN/negative_copy')

    # copy positive pictures as training set
    for i in range(ran_o):
        for filename in filelist_3:
            try:
                im = Image.open(filename)
                ss = filename[13:]
                im.save('CNN/positive_copy/' + ss[0:-4] + '_' + str(i) + '.jpg')
            except OSError as e:
                print(e.args)

    # copy negative pictures as a training set
    for filename in filelist_4:
        try:
            im = Image.open(filename)
            ss = filename[13:]
            im.save('CNN/negative_copy/' + ss[0:-4] + '.jpg')
        except OSError as e:
            print(e.args)

else:
    print('There is no need to update the model.')
