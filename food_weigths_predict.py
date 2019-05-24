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

import h5py


def get_model():
	base_model = InceptionV3(include_top=False)  # InceptionResNetV2(weights='imagenet', include_top=False)

	# add a global spatial averagedifferent types of data formats pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(2048, activation='relu')(x)
	x = Dense(2048, activation='relu')(x)
	x = Dense(2048, activation='relu')(x)
	x = Dense(2048, activation='relu')(x)

	predictions = Dense(1)(x)  # LeakyReLU

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)
	return model

model = get_model()

def read_test_csv(file_path:str):
	file = open(file_path)
	import csv
	lines = list(csv.reader(file))[1:]
	return lines

tests = read_test_csv("model_data/navn_vekt_fasit_val.csv")
model.load_weights("saves/model_after_first_fit.hdf5")
#test_image = "Dataset/fixed_images/Skumma_Kulturmjoe'lk_16_01.085.jpg"
for test_image, test_weigth in tests:
	img_path = "Dataset/fixed_images/"+test_image+".jpg"

	img = image.load_img(img_path, target_size=(299, 299))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	prediction = float(list(model.predict(x)[0])[0])
	print("predicted: {:<4.4f}, actual: {:<6.1f}, difference: {:<5.4f}, image: {:<40}".format(prediction, float(test_weigth), abs(prediction - float(test_weigth)), test_image))




