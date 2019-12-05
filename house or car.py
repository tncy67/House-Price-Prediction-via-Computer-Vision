#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Construct the path
DATA_PATH = "house_or_car_data"
TRAIN_PATH = '%s/train/' % (DATA_PATH)
VALID_PATH = '%s/valid/' % (DATA_PATH)
TEST_PATH = '%s/test/' % (DATA_PATH)


# In[2]:


from keras.preprocessing.image import ImageDataGenerator
image_width = 150
image_height = 150
image_size = (image_width, image_height)

batch_size = 20

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        TRAIN_PATH,
        # All images will be resized to 150x150
        target_size=image_size,
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        VALID_PATH,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

datagen= ImageDataGenerator(rescale=1./255)

data_generator = datagen.flow_from_directory(
        # This is the target directory
        TRAIN_PATH,
        # All images will be resized to 150x150
        target_size=image_size,
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

x, y = data_generator.next()

plt.figure(figsize=(16, 10))
for i, (img, label) in enumerate(zip(x, y)):
    plt.subplot(4, 5, i+1)
    if label == 1:
        plt.title('house')
    else:
        plt.title('car')
    plt.axis('off')
    plt.imshow(img, interpolation="nearest")


# In[5]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

lenet_model = Sequential()
lenet_model.add(Conv2D(6, (5, 5), activation='relu',name='conv1',
                        input_shape=(150, 150, 3)))
lenet_model.add(MaxPooling2D((2, 2), name='pool1'))
lenet_model.add(Conv2D(16, (5, 5), activation='relu', name='conv2'))
lenet_model.add(MaxPooling2D((2, 2), name='pool2'))
lenet_model.add(Flatten(name='flatten'))
lenet_model.add(Dense(120, activation='relu', name='fc1'))
lenet_model.add(Dense(84, activation='relu', name='fc2'))
lenet_model.add(Dense(1, activation='sigmoid', name='predictions'))


# In[6]:


lenet_model.summary()


# In[7]:


from keras.utils import plot_model


# In[8]:


import pydot
import pydotplus
import graphviz


# In[9]:


import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd


# In[10]:


plot_model(lenet_model, to_file='kitchen_and_livingRoom_lenet.png')


# In[11]:


SVG(model_to_dot(lenet_model, show_shapes=True).create(prog='dot', format='svg'))


# In[12]:


from keras import optimizers

lenet_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])


# In[13]:


from keras.callbacks import ModelCheckpoint, TensorBoard
import os

NAME = "lenet"
PATH = os.path.join('logs', NAME)
tensorboard = TensorBoard(log_dir=PATH)
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
best_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)



history = lenet_model.fit_generator(
        train_generator,
        steps_per_epoch=100, #2000/20 20 is the batch size
        validation_steps=8, #160/20
        epochs=10,
        validation_data=validation_generator,
        callbacks=[best_model, tensorboard])


# In[14]:


history_dict = history.history
print(history_dict.keys())


# In[15]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[16]:


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[17]:


from keras.preprocessing import image
import numpy as np
import os

fnames = [os.path.join(TEST_PATH, fname) for fname in os.listdir(TEST_PATH)]

# We pick one image to "augment"
img_path = fnames[5]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3) because the flow method requires the input array to be of rank 4
x = np.expand_dims(x, axis=0)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break


# In[18]:


train_datagen = ImageDataGenerator(
      rescale=1.0/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,  # this is the target directory
        target_size=image_size,  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255) # we only need to scale the input for validation set
validation_generator = validation_datagen.flow_from_directory(
        VALID_PATH,  # this is the target directory
        target_size=image_size,  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')


# In[19]:


from keras.layers import Dropout
lenet_model = Sequential()
lenet_model.add(Conv2D(6, (5, 5), activation='relu',name='conv1',
                        input_shape=(150, 150, 3)))
lenet_model.add(MaxPooling2D((2, 2), name='pool1'))
lenet_model.add(Conv2D(16, (5, 5), activation='relu', name='conv2'))
lenet_model.add(MaxPooling2D((2, 2), name='pool2'))
lenet_model.add(Flatten(name='flatten'))
# The new dropout layer
lenet_model.add(Dropout(0.5))
lenet_model.add(Dense(120, activation='relu', name='fc1'))
lenet_model.add(Dense(84, activation='relu', name='fc2'))
lenet_model.add(Dense(1, activation='sigmoid', name='predictions'))

lenet_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])


# In[20]:


from keras.callbacks import ModelCheckpoint, TensorBoard
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
best_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)

NAME = "lenet"
PATH = os.path.join('logs', NAME)
tensorboard = TensorBoard(log_dir=PATH)

history = lenet_model.fit_generator(
        train_generator,
        steps_per_epoch=100, #2000/20 20 is the batch size
        validation_steps=8, #160/20
        epochs=10,
        validation_data=validation_generator,
        callbacks=[best_model, tensorboard])


# In[21]:


history_dict = history.history
print(history_dict.keys())


# In[22]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[23]:


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# In[24]:


vgg_model = Sequential()
vgg_model.add(conv_base)
vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation='relu'))
vgg_model.add(Dense(1, activation='sigmoid'))


# In[25]:


vgg_model.summary()


# In[26]:


conv_base.trainable = False


# In[27]:


train_datagen = ImageDataGenerator(
      rescale=1.0/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,  # this is the target directory
        target_size=image_size,  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255) # we only need to scale the input for validation set
validation_generator = validation_datagen.flow_from_directory(
        VALID_PATH,  # this is the target directory
        target_size=image_size,  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')


# In[28]:


vgg_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
best_model = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)

NAME = "lenet"
PATH = os.path.join('logs', NAME)
tensorboard = TensorBoard(log_dir=PATH)

history = vgg_model.fit_generator(
            train_generator,
            steps_per_epoch=100, #2000/20 20 is the batch size
            validation_steps=8, #160/20
            epochs=2, # Change this to a bigger number if you want to train for more epochs
            validation_data=validation_generator,
            callbacks=[best_model, tensorboard])


# In[29]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[32]:


import os
test_iamges  = os.listdir(TEST_PATH)

from keras.preprocessing.image import load_img, img_to_array


def preprocess_image(img_path):
    img = load_img(img_path, target_size=image_size)
    img_tensor = img_to_array(img) 
    # change it to shape [1, 150, 150, 3]
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


plt.figure(figsize=(16, 12))
for index, image in enumerate(test_iamges):
    img = preprocess_image(TEST_PATH + image)
    
    prediction = vgg_model.predict(img)[0]
    
    plt.subplot(4, 5, index+1)
    if prediction < 0.5:
        plt.title('car %.2f%%' % (100 - prediction*100))
    else:
        plt.title('house %.2f%%' % (prediction*100))
    
    plt.axis('off')
    plt.imshow(img[0])


# In[33]:


img_tensor = preprocess_image(TEST_PATH + test_iamges[3])
plt.imshow(img_tensor[0])
plt.show()


# In[35]:


from keras.models import load_model

# model = load_model('./lenet/cats_and_dogs_lenet.h5')
model = load_model('./weights-improvement-09-0.98.hdf5')

model.summary()  # As a reminder.


# In[36]:


from keras import models

# Extracts the outputs of the top 4 layers:
layer_outputs = [layer.output for layer in model.layers[:4]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


# In[37]:


# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)


# In[38]:


first_layer_activation = activations[0]
print(first_layer_activation.shape)


# In[39]:


plt.figure(figsize=(16, 12))
for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(first_layer_activation[0, :, :, i])


# In[ ]:




