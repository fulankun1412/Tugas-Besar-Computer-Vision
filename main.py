# The main code

import cv2
import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras

# path to the dataset
paths = ['asl_dataset']
TOTAL_DATASET = 2515
x_train = []  # training lists
y_train = []
nb_classes = 36  # number of classes
img_rows, img_cols = 400, 400  # size of training images
img_channels = 3  # BGR channels
batch_size = 32
nb_epoch = 100  # iterations for training
data_augmentation = True

# dictionary for classes from char to numbers
classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}


# load the dataset and populate xtrain and ytrain
def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)
                    img = img_to_array(img)
                    x_train.append(img)
                    t = fullpath.rindex("/")
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex("/")
                    y_train.append(classes[fullpath[n + 1:t]])
                    




# create a model for training and return the model
def make_network(x_train):
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


# training model which was created
def train_model(model, X_train, Y_train):
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch)


# loads data set, converts the triaining arrays into required formats of numpy arrays and calls make_network to
# create a model and then calls train_model to train it and then saves the model in disk. OR just loads the model
# from disk.
def trainData():
    load_data_set()
    a = numpy.asarray(y_train)
    y_train_new = a.reshape(a.shape[0], 1)

    X_train = numpy.asarray(x_train).astype('float32')
    X_train = X_train / 255.0
    Y_train = np_utils.to_categorical(y_train_new, nb_classes)


    # run this if model is not saved.
    model = make_network(numpy.asarray(x_train))
    train_model(model,X_train,Y_train)
    model.save('keras.model')

    # run this if model is already saved on disk.
    # model = keras.models.load_model('/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/keras.model')

    return model


#model = trainData()
model = load_data_set()
x_train[1].shape

# called from main, when gesture is recognized. The gesture image is cropped and sent to this function.
def identifyGesture(handTrainImage):
    # saving the sent image for checking
    # cv2.imwrite("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a0.jpeg", handTrainImage)

    # converting the image to same resolution as training data by padding to reach 1:1 aspect ration and then
    # resizing to 400 x 400. Same is done with training data in preprocess_image.py. Opencv image is first
    # converted to Pillow image to do this.
    handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(handTrainImage)
    img_w, img_h = img.size
    M = max(img_w, img_h)
    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(img, offset)
    size = 400,400
    background = background.resize(size, Image.ANTIALIAS)

    # saving the processed image for checking.
    # background.save("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a.jpeg")

    # get image as numpy array and predict using model
    open_cv_image = numpy.array(background)
    background = open_cv_image.astype('float32')
    background = background / 255
    background = background.reshape((1,) + background.shape)
    predictions = model.predict_classes(background)

    # print predicted class and get the class name (character name) for the given class number and return it
    print(predictions)
    key = (key for key, value in classes.items() if value == predictions[0]).next()
    return key


def nothing(x):
    pass


