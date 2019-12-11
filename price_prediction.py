from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import concatenate
import argparse
import locale
import imutils


trainY = trainAttrX["price"]
testY = testAttrX["price"]

(trainAttrX, testAttrX) = datasets.house_numerical(df,trainAttrX, testAttrX)

mlp = models.mlp(trainAttrX.shape[1], regress=False)
cnn = models.cnn(208, 156, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)
x = Dense(3, activation="softmax")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# mean_absolute_percentage_error
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=5, batch_size=8)

preds = model.predict([testAttrX, testImagesX])

import matplotlib.pyplot as plt
correct_indices = [10,11,12,13,14,15,16,17,18,19]
plt.figure(figsize=(12, 9))
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(testImagesX[i])
#     plt.title("{}".format(((preds[i].max())*100)))
    if preds[i].max() > 0.3:
        if preds[i].max()==preds[i][0]:
            print("prediction is low")
        if preds[i].max()==preds[i][1]:
            print("prediction is medium")
        if preds[i].max()==preds[i][2]:
            print("prediction is high")
        if testY.iloc[i]==2:
            print("it is in fact high")
        if testY.iloc[i]==1:
            print("it is in fact medium")
        if testY.iloc[i]==0:
            print("it is in fact low")
