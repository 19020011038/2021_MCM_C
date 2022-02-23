import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.inf)

# import the model
model = tf.keras.models.load_model('my_model.h5')
model.summary()

print("finish loading model！")
dict_label = {0: 'positive', 1: 'negative'}


def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


def im_xiangsu(paths):
    filelist_temp = []
    for filename in paths:
        try:
            im = Image.open(filename)
            newim = im.resize((128, 128))
            filelist_temp.append(newim)
        except OSError as e:
            print(e.args)
    return filelist_temp


# image data into an array
def im_array(paths):
    M = []
    for filename in paths:
        im = filename
        im_L = im.convert("L")  # Pattern L
        Core = im_L.getdata()
        arr1 = np.array(Core, dtype='float32') / 255.0
        list_img = arr1.tolist()
        M.extend(list_img)
    return M


path_1 = 'CNN/positive/'
path_2 = 'CNN/negative/'
path_3 = 'CNN/unverified/'
filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)
filelist_3 = read_image(path_3)


def getresult(paths):

    filelist = read_image(paths)
    filelist_test = im_xiangsu(filelist)
    img = im_array(filelist_test)
    test_images = np.array(img).reshape(len(filelist_test), 128, 128)
    test_images = test_images[..., np.newaxis]
    predictions_single = model.predict(test_images)

    print(predictions_single)
    positive_number = 0
    negative_number = 0
    rate_sure = 0.0

    for i in range(len(filelist)):
        filename = filelist[i]
        if predictions_single[i][0] >= predictions_single[i][1]:
            positive_number += 1
            rate_sure += predictions_single[i][0] - predictions_single[i][1]
            im = Image.open(filelist[i])
            ss = filename[15:]
            im.save('CNN/positive/' + ss)
            # im.save('CNN/positive_copy/' + ss)

        else:
            negative_number += 1
            rate_sure += predictions_single[i][1] - predictions_single[i][0]
            im = Image.open(filelist[i])
            ss = filename[15:]
            im.save('CNN/negative/' + ss)
            # im.save('CNN/negative_copy/' + ss)

    rate_sure_average = rate_sure / len(predictions_single)  # the rate of negative

    return positive_number, negative_number, rate_sure_average


# do predict
exist_positive = len(filelist_1)
exist_negative = len(filelist_2)
predict_positive_number, predict_negative_number, predict_rate = getresult('CNN/unverified/')
all_negative = predict_negative_number + exist_negative
all_number = predict_positive_number + predict_negative_number + exist_positive + exist_negative
rate_final = all_negative / all_number
rate_accuracy_1 = (exist_positive + exist_negative) * 1.0
rate_accuracy_2 = (predict_positive_number + predict_negative_number) * predict_rate
rate_accuracy = rate_accuracy_2 / (predict_positive_number + predict_negative_number)

print(predict_positive_number)
print(predict_negative_number)
print(exist_positive)
print(exist_negative)
print('The total numbers of negative id is :' + str(all_negative))
print('The rate of the mistaken classification is : ' + str(rate_final * 100) + '%')
print('The accuracy of the model is : ' + str(rate_accuracy * 100) + '%')


