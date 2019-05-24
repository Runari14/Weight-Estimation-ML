from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
import  keras.applications

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
import os

import random
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from keras.layers import LeakyReLU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'

Target_directory = 'Dataset/seperated_food_images/classes/bread'
#n_pictures = 101000
batch_size = 6


early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/-{epoch:02d}-{loss:.2f}', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# create the base pre-trained model
base_model = InceptionV3(include_top=False)#InceptionResNetV2(weights='imagenet', include_top=False)

# add a global spatial averagedifferent types of data formats pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(2048, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1)(x) # LeakyReLU

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mean_squared_error')


def append_ext(fn):
    return fn + ".jpg"


def toint(fn):
    return int(fn)

traindf = pd.read_csv("model_data/navn_vekt_fasit.csv", dtype = str)
testdf = pd.read_csv("model_data/navn_vekt_fasit_val.csv", dtype=str)

traindf["id"] = traindf["id"].apply(append_ext)
testdf["id"] = testdf["id"].apply(append_ext)
traindf["label"] = traindf["label"].apply(toint)
testdf["label"] = testdf["label"].apply(toint)


#print(traindf)
#print(traindf[traindf.index.duplicated()])
#print("test",testdf[testdf.index.duplicated()])
datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.05)
#print("\n"*5)

train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=Target_directory,
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=10,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(299, 299))

valid_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=Target_directory,
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=10,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(299, 299))

test_datagen = ImageDataGenerator(rescale=1. / 255.)
"""
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="Dataset/fixed_images",
    x_col="id",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(299, 299))"""


ckpt = keras.callbacks.ModelCheckpoint(
    "model/first-{epoch:02d}-{loss:.2f}",
    monitor='loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    period=1)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
#STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
#print(STEP_SIZE_TEST)
if STEP_SIZE_VALID == 0:
    STEP_SIZE_VALID += 1
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    use_multiprocessing=True,
    epochs=30,
    callbacks=[ckpt])




print("starting first train")
# train the model on the new data for a few epochs
#model.fit_generator(train_generator, n_pictures//batch_size, 3 , callbacks=[ckpt, early, tensorboard])
model.save("saves/model_after_first_fit.hdf5")
print("saved first")
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
   #print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='mean_squared_error')



# this is a similar generator, for validation data
#validation_generator = test_datagen.flow_from_directory(
#        'data/validation',
#        target_size=(150, 150),
#        batch_size=batch_size,
#        class_mode='binary')

# callbacks
ckpt = keras.callbacks.ModelCheckpoint("model/last-{epoch:02d}-{loss:.2f}", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

print("start second train")
model.fit_generator(train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_data = valid_generator,
    validation_steps = STEP_SIZE_VALID,
    use_multiprocessing = True,
    epochs = 60,
    callbacks=[ckpt, tensorboard])


model.save("saves/model_final_weigths_old.hdf5")
print("finished successfylly!! wow!! yey!!")