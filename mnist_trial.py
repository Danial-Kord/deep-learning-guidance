import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.layers import Flatten, Dense, Conv2D,MaxPooling2D


def train_and_test(x_train,y_train,x_test,y_test):
    model = keras.models.Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation=keras.activations.relu))
    model.add(Dense(128, activation=keras.activations.relu))
    model.add(Dense(10, activation=keras.activations.softmax))

    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)

    loss, acc = model.evaluate(x_test, y_test)

    print("final evaluation accuracy", acc)

def train_and_test2(x_train,y_train,x_test,y_test):
    x_train = np.array(x_train).reshape(-1,28,28,1)
    x_test = np.array(x_test).reshape(-1,28,28,1)

    model = keras.models.Sequential()

    model.add(Conv2D(128,(3,3),input_shape=x_train.shape[1:],activation=keras.activations.relu))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation=keras.activations.relu))
    model.add(Dense(128, activation=keras.activations.relu))
    model.add(Dense(10, activation=keras.activations.softmax))

    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3,batch_size=32)

    loss, acc = model.evaluate(x_test, y_test)

    print("final evaluation accuracy", acc)



data_set = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = data_set.load_data()



print("results without normalization preprocessing : ")
train_and_test(x_train,y_train,x_test,y_test)
print("-----------------------")

x_train = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)

print("results with normalization preprocessing : ")
train_and_test(x_train,y_train,x_test,y_test)


print("results using normalization preprocessing + CONV layer : ")
train_and_test2(x_train,y_train,x_test,y_test)
print("-----------------------")