import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
np.random.seed(2)
dataset = pd.read_csv("train_final.csv")
y = dataset["784"]
X = dataset.drop(labels = ["784"], axis = 1)
X = X / 255.0
X = X.values.reshape(-1,28,28,1)
y = to_categorical(y, num_classes = 14)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2, stratify = y)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(14, activation = "softmax"))
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

model.fit_generator(datagen.flow(X_train,y_train, batch_size=60),epochs = 10,validation_data = (X_train,y_train), verbose = 1, steps_per_epoch=X_train.shape[0] // 60)
model.save("pro1.h5")