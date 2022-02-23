import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras import regularizers
from keras.layers.core import Dropout


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


filelist_all = im_xiangsu(filelist_1) + im_xiangsu(filelist_2)


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


M = im_array(filelist_all)

# prepare training data
dict_label = {0: 'positive', 1: 'negative'}
train_images = np.array(M).reshape(len(filelist_all), 128, 128)
train_images = train_images[..., np.newaxis]

label = [0] * len(filelist_1) + [1] * len(filelist_2)
train_lables = np.array(label)

print('_______________________')
print(train_images.shape)
print(train_lables.shape)

# construct neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1),
                        kernel_regularizer=regularizers.l2(0.01)))

# set pooling layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))


# reduce dimension
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.75))
model.add(layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))


# show the structure of the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# set training times
model.fit(train_images, train_lables, epochs=2, validation_split=0.2)

# save the model as pd
model.save('my_model.h5')
print(model.summary())
tf.keras.models.save_model(model, 'CNN/models')
print("save model success！")
