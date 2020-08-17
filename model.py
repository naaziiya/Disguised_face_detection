#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import glob  # Extract specific files
import os
import pandas as pd
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
from skimage.io import imread


# # 1. Data Preparation

# ### Load Data

# In[ ]:


path = './disguise_face_dataset'


# ### PENOMONIA vs NORMAL

# # 2. Data Augmentation

# In[ ]:


train_dir = './disguise_face_dataset'
train_datagen = ImageDataGenerator(
    rescale=1./255,
)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # image size
    batch_size=2
)


# ### Apply for train set

# In[ ]:


nb_train_samples = 100
img_width, img_height = 160, 160
batch_size = 8


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
)


# # 3. Modeling

# In[ ]:


###### VGG16 ######
def vgg_m():
    input_data = Input(shape=(img_width, img_height, 3), name="InputData")

    # (1)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', activation='relu')(input_data)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # (2)
    x = Conv2D(filters=128, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # (3)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # (4)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # (5)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # (6)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(input_data, output)

    return model


# In[ ]:


vgg = vgg_m()
vgg.summary()


# In[ ]:


# When I used Adam, the model fell into the local minima
vgg.compile(optimizer=RMSprop(lr=0.00005),
            loss='categorical_crossentropy', metrics=['acc'])


# In[ ]:


history = vgg.fit_generator(training_set,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=120)

# vgg_model.save('D:/chest-xray/model/'+'term_prj.h5')


# # 4. Evaluation

# In[ ]:


vgg.save('model.h5')


# In[ ]:


acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Loss')
plt.legend()
plt.show()


# In[ ]:
