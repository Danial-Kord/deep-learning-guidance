import time

import numpy
import tensorflow as tf
from tensorflow.python.saved_model import builder as pb_builder
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
import cv2
import os
import random
import pickle

IMG_SIZE = 50
DIR = "E:\Machine learning data\Kaggle cats and dogs\PetImages" #Kaggle (dog and cat) images path
Categorries = ["Dog", "Cat"]






def collect_data():
    x_data = []
    y_data = []
    data = []
    for category in Categorries:
        path = os.path.join(DIR,category)
        for image in os.listdir(path):
            try:
                image_data = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
                image_data = cv2.resize(image_data,(IMG_SIZE,IMG_SIZE))
                # data.append([image_data,Categorries.index(category)])
                x_data.append(image_data)
                y_data.append(int(Categorries.index(category)))
            except:
                print("wrong image data: ",os.path.join(path,image))


    joined_lists = list(zip(x_data, y_data))
    random.shuffle(joined_lists) # shuffle foe better learning(so all same categories wont be near together in array)
    x_data, y_data = zip(*joined_lists)
    # random.shuffle(data)
    # for image, num in data:
    #     x_data.append(image)
    #     y_data.append(num)
    print(len(y_data), len(x_data))
    plt.imshow(x_data[0])
    plt.show()
    print(Categorries[y_data[0]])
    return x_data,y_data
def save_data(data,name):
    pickle_out = open(name + ".pickle","wb")
    pickle.dump(data,pickle_out)
    pickle_out.close()

def load_data(name):
    pickle_out = open(name + ".pickle","rb")
    return pickle.load(pickle_out)

option = int(input("enter 1 for loading all images from hard or 2 for reading loaded images from before or 3 for using pre-trained model"))

if option == 1:

    x_data,y_data = collect_data()

    x_data = np.array(x_data).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    y_data = np.array(y_data)
    save_data(x_data,"kaggle\\X")
    save_data(y_data,"kaggle\\Y")
else:
    x_data = load_data("kaggle\\X")
    y_data = load_data("kaggle\\Y")

# TRAINING MODEL

# hyper parameters
CNN_layers = [3,4]
dense_layers = [0,1,2]
dense_layer_sizes = [64,128]
CNN_layer_sizes = [64,128]

best_accuracy = 0
best_model = None
best_model_name = ""

print(x_data.shape[1:])
for CNN_layer in CNN_layers:
    for dense_layer in dense_layers:
        for CNN_layer_size in CNN_layer_sizes:
            for dense_layer_size in dense_layer_sizes:
                cat_dog_model_name = 'CatDog{}-{}CNN-{}Size-{}Dense-{}Size'.format(str(int(time.time())),CNN_layer,CNN_layer_size,dense_layer,dense_layer_size)
                board = TensorBoard(log_dir='logs/'+cat_dog_model_name)

                model = keras.models.Sequential()

                model.add(Conv2D(CNN_layer_size,(3,3),input_shape=x_data.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

                for i in range(CNN_layer-1):
                    model.add(Conv2D(CNN_layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for i in range(dense_layer):
                    model.add(Dense(dense_layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(int(len(Categorries)), activation=keras.activations.softmax))

                model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
                model.fit(x_data,y_data,batch_size=32,epochs=10,validation_split=0.25,callbacks=[board])
                loss, acc = model.evaluate(x_data, y_data)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_model_name = cat_dog_model_name

best_model.save('path/to/your/{}.md5'.format(best_model_name))











