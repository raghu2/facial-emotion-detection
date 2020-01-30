import numpy as np
import pandas as pd
import face_recognition_models
import cv2
import os
import keras
from keras import models
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow

save_dir = os.path.join(os.getcwd(), 'saved_models')

data = pd.read_csv("fer2013.csv")
abc = data.values
labels = np.array(abc[:,0])
pics = abc[:,1:2]
b = [] * len(labels)

for i in range(len(labels)):
    pics[i][0] = pics[i][0].split(' ')
    pics[i][0] = list(map(int, pics[i][0]))
    pics[i][0] = np.array(pics[i][0])
    b.append(np.reshape(pics[i][0], (48, 48)))

c = np.array(b)

x_train = c[:16000]
x_valid = c[13000:16000]
x_test = c[32000:]

y_train = labels[:16000]
y_valid = labels[13000:16000]
y_test = labels[32000:]


batch_size = 64
num_classes = 7
epochs = 25
learn_rate = 0.01
decay_rate = 1e-5

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#print(x_train.shape)

x_train = x_train.reshape(16000, 48, 48, 1)
x_valid = x_valid.reshape(3000, 48, 48, 1)
x_test = x_test.reshape(3887, 48, 48, 1)

x_train = x_train/255
x_valid = x_valid/255
x_test = x_test/255

model_name = "face_detection_model6.h5" # make this 6

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.4)) # make this a comment

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.6)) # make this a comment

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(7 , activation='softmax'))

#opt = keras.optimizers.rmsprop(learn_rate, decay_rate)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size, epochs, validation_data = (x_valid, y_valid), shuffle=True)

scores = model.evaluate(x_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)