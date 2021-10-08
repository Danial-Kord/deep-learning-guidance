import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

def train_and_test(x_train,y_train,x_test,y_test):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=keras.activations.relu))
    model.add(keras.layers.Dense(128, activation=keras.activations.relu))
    model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)

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
