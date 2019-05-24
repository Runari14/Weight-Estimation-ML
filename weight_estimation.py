import random
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
#from shutil import copy
#from keras.layers import LeakyReLU


def blackout(img, xmin:int, ymin:int, xmax:int, ymax:int):
	img_array = np.array(img)
	img_array[ymin:ymax, xmin:xmax] = 0
	img = Image.fromarray(img_array)
	return img


def extract(img, xmin:int, ymin:int, xmax:int, ymax:int):
	img_array = np.array(img)
	img_array = img_array[ymin:ymax, xmin:xmax]
	img = Image.fromarray(img_array)
	img.show()
	return img


def min_max_average(predictions:list):
	return min(predictions), max(predictions), sum(predictions)/len(predictions)


def loop_files(source_dir:str, save_loaction:str):
	amount = len(os.listdir(source_dir)) / 2
	finished_with = 0
	for file in os.listdir(source_dir):
		if file.endswith(".xml"):
			f = open(source_dir+"/"+file)
			file_path = source_dir+"/"+file.replace(".xml", ".jpg")
			if os.path.isfile(file_path):
				img = Image.open(file_path)
			else:
				file_path = file_path.replace("oe", "oe'")
				file_path = file_path.replace("aa", "aa'")
				file_path = file_path.replace("ae", "ae'")
				file_path = file_path.replace("musli", "mu'sli")
				img = Image.open(file_path)

			tree = ET.parse(f)
			root = tree.getroot()
			for obj in root.iter('object'):
				class_name = obj.find('name').text
				if class_name == "post-it" or class_name == "weight":
					xmlbox = obj.find('bndbox')
					#img = blackout(img, int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
			        #    int(xmlbox.find('ymax').text))
					img = blackout(
						img,
						int(xmlbox.find('xmin').text),
						int(xmlbox.find('ymin').text),
						int(xmlbox.find('xmax').text),
						int(xmlbox.find('ymax').text))
			img.save(save_loaction + file.replace(".xml", ".jpg"))
			img.close()
			f.close()
			finished_with += 1


			print("{}/{}, {}%".format(finished_with, amount, finished_with/amount*100))


def read_xslx(xl_path: str, save_path: str = None):
	xl = pd.read_excel(xl_path, usecols=["Bildenavn", "Vekt"])
	xl_nans = xl.notna()
	output = []

	names = []

	for i in range(len(xl["Bildenavn"])):
		if xl_nans["Bildenavn"][i]: # if the sheet has values in the field. if not it is not a image and just a empty line.
			name: str = xl["Bildenavn"][i]
			name = name.replace(" ", "_")
			name = name.replace(",", "")
			name = name.replace("æ", "ae")
			name = name.replace("ø", "oe")
			name = name.replace("å", "aa")
			name = name.replace("ü", "u")
			if name not in names:
				output.append([name, xl["Vekt"][i]])
				names.append(name)
			else:
				print(name)
	output.sort()
	#from pprint import pprint
	#pprint(output)

	if save_path:
		f = open(save_path, "w+")
		f2 = open(save_path.replace(".csv", "_val.csv"), "w+")

		f.write("id,label\n")
		f2.write("id,label\n")

		random.seed(1)
		random.shuffle(output)
		length = len(output)
		#if length > 100:
		#	val_amount = (length // 100) * 5
		#else:
		#	val_amount = 5
		#print(length)
		for name, weight in output:
			f.write(str(name) + "," + str(weight) + "\n")
			#if val_amount == 0:
			#	f.write(str(name) + "," + str(weight) + "\n")
			#else:
			#	val_amount -= 1
			#	f2.write(str(name)+","+str(weight)+"\n")
		f.close()
		f2.close()

	return output
def extract_test_data():

	bread = os.listdir("Dataset/seperated_food_images/extraced_classes/bread")
	cheese = os.listdir("Dataset/seperated_food_images/extraced_classes/cheese")
	cmilk = os.listdir("Dataset/seperated_food_images/extraced_classes/chokolate_milk")
	crisp = os.listdir("Dataset/seperated_food_images/extraced_classes/crispbread")
	milk = os.listdir("Dataset/seperated_food_images/extraced_classes/milk")
	yog = os.listdir("Dataset/seperated_food_images/extraced_classes/yoghurt")

	#traindf = pd.read_csv("model_data/navn_vekt_fasit_2_0.csv", dtype=str)
	f = open("model_data/navn_vekt_fasit_2_0.csv")
	lines = f.readlines()
	f.close()
	test_data = []
	data = []

	counter_bread = 0
	counter_cheese = 0
	counter_cmilk = 0
	counter_crisp = 0
	counter_milk = 0
	counter_yog = 0

	for line in lines:
		img_path = line.split(",")[0]+ ".jpg"
		if img_path in bread:
			if len(bread)/100 * 10 <= counter_bread: # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_bread += 1

		elif img_path in cheese:
			if len(cheese) / 100 * 10 <= counter_cheese:  # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_cheese += 1

		elif img_path in cmilk:
			if len(cmilk) / 100 * 10 <= counter_cmilk:  # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_cmilk += 1

		elif img_path in crisp:
			if len(crisp) / 100 * 10 <= counter_crisp:  # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_crisp += 1

		elif img_path in milk:
			if len(milk) / 100 * 10 <= counter_milk:  # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_milk += 1

		elif img_path in yog:
			if len(yog) / 100 * 10 <= counter_yog:  # all elements that should go in data.
				data.append(line)
			else:
				test_data.append(line)
			counter_yog += 1

	f = open("model_data/navn_vekt_fasit_2_0.csv", "w")
	f.write("id,label\n")
	f.writelines(data)
	f.close()

	f = open("model_data/navn_vekt_fasit_2_0_val.csv", "w")
	f.write("id,label\n")
	f.writelines(test_data)
	f.close()
	        	

from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers



#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1.0
#set_session(tf.Session(config=config))


def model(is_large=True) -> Sequential:
	model = Sequential()
	model.add(Conv2D(64, (3, 3), padding='same',
					 input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Activation('relu'))
	if is_large:
		model.add(Conv2D(256, (3, 3)))
		model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(Activation('relu'))
	if is_large:
		model.add(Dropout(0.2))
		model.add(Dense(2048))
		model.add(Activation('relu'))

	model.add(Dense(1))
	#model.add(LeakyReLU(alpha=0.1))
	model.compile(optimizers.Adam(lr=0.0001), loss="mean_squared_error", metrics=["accuracy"])#rmsprop(lr=0.000_01, decay=1e-6), loss="mean_squared_error", metrics=["accuracy"])
	return model

def train(model:Sequential, data_directory:str, csv_base_name:str):

	def append_ext(fn):
		return fn + ".jpg"

	def toint(fn):
		return int(fn)

	traindf = pd.read_csv(csv_base_name+".csv", dtype=str)
	testdf = pd.read_csv(csv_base_name+"_val.csv", dtype=str)
	#traindf = pd.read_csv("model_data/navn_vekt_fasit.csv", dtype = str)
	#testdf = pd.read_csv("model_data/navn_vekt_fasit_val.csv", dtype=str)

	traindf["id"] = traindf["id"].apply(append_ext)
	testdf["id"] = testdf["id"].apply(append_ext)
	traindf["label"] = traindf["label"].apply(toint)
	testdf["label"] = testdf["label"].apply(toint)

	target_size = (150, 150)
	batch_size = 5

	#print(traindf)
	#print(traindf[traindf.index.duplicated()])
	#print("test",testdf[testdf.index.duplicated()])
	datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.1)
	#print("\n"*5)

	train_generator = datagen.flow_from_dataframe(
		dataframe=traindf,
		directory=data_directory,
		x_col="id",
		y_col="label",
		subset="training",
		batch_size=batch_size,
		seed=42,
		shuffle=True,
		class_mode="other",
		target_size=target_size)

	valid_generator = datagen.flow_from_dataframe(
		dataframe=traindf,
		directory=data_directory,
		x_col="id",
		y_col="label",
		subset="validation",
		batch_size=batch_size,
		seed=42,
		shuffle=True,
		class_mode="other",
		target_size=target_size)

	test_datagen = ImageDataGenerator(rescale=1. / 255.)

	test_generator = test_datagen.flow_from_dataframe(
		dataframe=testdf,
		directory=data_directory,
		x_col="id",
		y_col=None,
		batch_size=batch_size,
		seed=42,
		shuffle=False,
		class_mode=None,
		target_size=target_size)



	step_size_train = train_generator.n // train_generator.batch_size
	step_size_valid = valid_generator.n // valid_generator.batch_size
	step_size_test = test_generator.n // test_generator.batch_size
	print(step_size_test)
	if step_size_valid == 0:
		step_size_valid += 1
	model.fit_generator(generator=train_generator,
						steps_per_epoch=step_size_train,
						validation_data=valid_generator,
						validation_steps=step_size_valid,
						use_multiprocessing=True,
						epochs=30
						)
	names = test_generator.filenames
	predictions = model.predict_generator(
		generator=test_generator,
		steps=1)
	predictions = list(map(lambda x: x[0], predictions))
	labeled_predicitons = list(zip(names, predictions))

	for label, prediction in labeled_predicitons:
		print("{:4.4f}, name: {}".format(prediction, label))

	return model

def predict(model:Sequential, data_directory:str, source_csv):

	def append_ext(fn):
		return fn + ".jpg"

	def toint(fn):
		return int(fn)

	target_size = (150, 150)
	batch_size = 6
	test_datagen = ImageDataGenerator(rescale=1. / 255.)

	testdf = pd.read_csv(source_csv, dtype=str)
	testdf["id"] = testdf["id"].apply(append_ext)
	testdf["label"] = testdf["label"].apply(toint)
	print(testdf)
	test_generator = test_datagen.flow_from_dataframe(
		dataframe=testdf,
		directory=data_directory,
		x_col="id",
		y_col=None,
		batch_size=batch_size,
		seed=42,
		shuffle=False,
		class_mode=None,
		target_size=target_size)

	step_size_test = test_generator.n // test_generator.batch_size
	print(step_size_test)
	if step_size_test == 0:
		step_size_test += 1

	names = test_generator.filenames


	#testdf.index[testdf['id'] == name].tolist()


	predictions = model.predict_generator(
		generator=test_generator,
		steps=step_size_test)
	predictions = list(map(lambda x: x[0], predictions))
	labeled_predicitons = list(zip(names, predictions))
	#print(testdf[names[0]])
	i = 0

	original_weigths = []
	diff_from_original = []
	for label, prediction in labeled_predicitons:

		index = testdf.index[testdf['id'] == label].tolist()[0]

		print("{:4.4f}, name: {}, original_weigth: {} diff: {}".format(prediction, label, testdf["label"][index], prediction - testdf["label"][index]))
		original_weigths.append(testdf["label"][index])
		diff_from_original.append(((prediction - testdf["label"][index])**2)**0.5)
		i += 1
	print("\n")

	from matplotlib import pyplot as plt

	plt.rcParams.update({'font.size': 14})
	plt.figure(figsize=(10,7))
	plt.hist(diff_from_original,rwidth=0.9)
	plt.suptitle('Milk Training set', fontsize=20)
	plt.xlabel('Difference', fontsize=18)
	plt.ylabel('Frequency', fontsize=16)
	plt.show()

	print("average_distance: {}, min distance:{}, max Distance: {}".format(sum(diff_from_original)/len(diff_from_original), min(diff_from_original), max(diff_from_original)))
	print("min prediction: {:<3.4f}, max prediction = {:<3.4f}, average prediction: {:<3.4f}".format(*min_max_average(predictions)))

	print("min original: {:<3.4f}, max original = {:<3.4f}, average original: {:<3.4f}".format(*min_max_average(original_weigths)))


def load_dataframe(path) -> pd.DataFrame:
	def append_ext(fn):
		return fn + ".jpg"

	def toint(fn):
		return int(fn)
	
	dataframe = pd.read_csv(path, dtype=str)
	dataframe["id"] = dataframe["id"].apply(append_ext)
	dataframe["label"] = dataframe["label"].apply(toint)
	
	return dataframe


def create_datagenerator(df:pd.DataFrame, data_path:str):
	target_size = (150, 150)
	batch_size = 6
	test_datagen = ImageDataGenerator(rescale=1. / 255.)
	test_generator = test_datagen.flow_from_dataframe(
		dataframe=df,
		directory=data_path,
		x_col="id",
		y_col=None,
		batch_size=batch_size,
		seed=42,
		shuffle=False,
		class_mode=None,
		target_size=target_size)
	return test_generator


def calculate_diff(generator, df:pd.DataFrame, model:Sequential):
	step_size = 1
	names = generator.filenames

	# testdf.index[testdf['id'] == name].tolist()

	predictions = model.predict_generator(
		generator=generator,
		steps=step_size)
	predictions = list(map(lambda x: x[0], predictions))
	labeled_predicitons = list(zip(names, predictions))
	# print(testdf[names[0]])
	i = 0

	original_weigths = []
	diff_from_original = []
	for label, prediction in labeled_predicitons:
		index = df.index[df['id'] == label].tolist()[0]

		#print("{:4.4f}, name: {}, original_weigth: {} diff: {}".format(prediction, label, testdf["label"][index],
		#                                                               prediction - testdf["label"][index]))
		original_weigths.append(df["label"][index])
		diff_from_original.append(((prediction - df["label"][index]) ** 2) ** 0.5)
		i += 1
	#print("\n")
	return diff_from_original


def standard_error():
	base_model_path = "model_data/trained_weights_/"
	model_path_bread = base_model_path+"Bread.hdf5"
	model_path_crisp = base_model_path+"crispbread.hdf5"
	model_path_cheese = base_model_path+"cheese.hdf5"
	model_path_milk = base_model_path+"milk.hdf5"
	model_path_chomilk = base_model_path+"chomilk.hdf5"
	model_path_yogurt = base_model_path+"yogurt.hdf5"
	
	base_dir_path = "Dataset/seperated_food_images/extraced_classes/"
	dir_bread = base_dir_path+"bread"
	dir_crisp = base_dir_path+"crispbread"
	dir_cheese = base_dir_path+"cheese"
	dir_milk = base_dir_path+"milk"
	dir_chomilk = base_dir_path+"chokolate_milk"
	dir_yogurt = base_dir_path+"yoghurt"
	

	bread = model(False)
	bread.load_weights(model_path_bread)
	crisp = model()
	crisp.load_weights(model_path_crisp)
	cheese = model()
	cheese.load_weights(model_path_cheese)
	milk = model()
	milk.load_weights(model_path_milk)
	chomilk = model()
	chomilk.load_weights(model_path_chomilk)
	yogurt = model()
	yogurt.load_weights(model_path_yogurt)
	
	# df contains data for all datasets, however it will only load them if they are avaible in the testing directory.
	df = load_dataframe("model_data/navn_vekt_fasit_2_0_val.csv")

	gen_bread = create_datagenerator(df, dir_bread)
	gen_crisp = create_datagenerator(df, dir_crisp)
	gen_cheese = create_datagenerator(df, dir_cheese)
	gen_milk = create_datagenerator(df, dir_milk)
	gen_chomilk = create_datagenerator(df, dir_chomilk)
	gen_yogurt = create_datagenerator(df, dir_yogurt)

	diff_bread = calculate_diff(gen_bread, df, bread)
	diff_crisp = calculate_diff(gen_crisp, df, crisp)
	diff_cheese = calculate_diff(gen_cheese, df, cheese)
	diff_milk = calculate_diff(gen_milk, df, milk)
	diff_chomilk = calculate_diff(gen_chomilk, df, chomilk)
	diff_yogurt = calculate_diff(gen_yogurt, df, yogurt)

	avg_bread = sum(diff_bread)     /len(diff_bread)
	avg_crisp = sum(diff_crisp)     / len(diff_crisp)
	avg_cheese = sum(diff_cheese)   / len(diff_cheese)
	avg_milk = sum(diff_milk)       / len(diff_milk)
	avg_chomilk = sum(diff_chomilk) / len(diff_chomilk)
	avg_yogurt = sum(diff_yogurt)   / len(diff_yogurt)

	sum_bread = sum(diff_bread)
	sum_crisp = sum(diff_crisp)
	sum_cheese = sum(diff_cheese)
	sum_milk = sum(diff_milk)
	sum_chomilk = sum(diff_chomilk)
	sum_yogurt = sum(diff_yogurt)
	
	#total_sum = sum_bread #+ sum_cheese + sum_crisp + sum_yogurt #+ sum_chomilk  + sum_milk
	#total_num_samples = +len(diff_bread) #+ len(diff_crisp)+ len(diff_cheese) + len(diff_yogurt) #+ len(diff_milk) + len(diff_chomilk)
	#total_mean = total_sum/total_num_samples
	# mean for all samples total:
	def standard_error(inn):
		total_sum = sum(inn)
		total_num_samples = len(inn)
		total_mean = total_sum / total_num_samples
		s = 0
		for value_lists in [inn]:#, diff_crisp, diff_cheese, diff_yogurt]:#, diff_milk, diff_chomilk]:
			for value in value_lists:
				s += (value - total_mean)**2

		s = (((1/total_num_samples) * s)**0.5)/ (total_num_samples)**0.5
		return s
	print("milk:",standard_error(diff_milk))
	print("bread:", standard_error(diff_bread))
	print("yogurt:", standard_error(diff_yogurt))
	print("crispbread:", standard_error(diff_crisp))
	print("chomilk:", standard_error(diff_chomilk))
	print("cheese:", standard_error(diff_cheese))
	s = "s"
	return s
	
#img = np.array(Image.open("Dataset/Bilder/Appelsinjuice_2_06.568.jpg"))
	#img.reshape((1, 3024, 4032, 3))
	#print(model.predict([img]))
#read_xslx("model_data/AI matvarebilde og næringsinnhold - Updated Margo Van H.xlsx", "model_data/navn_vekt_fasit_2_0.csv")
#extract_test_data()
#print(standard_error())

model = model()
model.load_weights("model_data/trained_weights_/milk.hdf5")
predict(model, "Dataset/seperated_food_images/extraced_classes/milk", "model_data/navn_vekt_fasit_2_0.csv")
#cmilk = os.listdir("Dataset/seperated_food_images/extraced_classes/chokolate_milk")
#milk = os.listdir("Dataset/seperated_food_images/extraced_classes/milk")

#print("starting with bread")
#trained_model = train(model(), "Dataset/seperated_food_images/classes/bread","model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/Bread_whole_images.hdf5")


#model.load_weights("model_data/trained_weights_/chomilk.hdf5")
#predict(model, "Dataset/seperated_food_images/extraced_classes/chokolate_milk", "model_data/navn_vekt_fasit_2_0.csv")
#predict(model, "Dataset/seperated_food_images/extraced_classes/chokolate_milk", "model_data/navn_vekt_fasit_2_0_val.csv")
#model = model()
#model.load_weights("model_data/trained_weights_/Bread.hdf5")


#print("starting with cheese")
#trained_model = train(model(), "Dataset/seperated_food_images/extraced_classes/cheese","model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/cheese.hdf5")
#del trained_model
#print("starting with crispbread")
#trained_model = train(model(), "Dataset/seperated_food_images/extraced_classes/crispbread", "model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/crispbread.hdf5")
#del trained_model
#print("starting with chocolate milk")
#trained_model = train(model(), "Dataset/seperated_food_images/extraced_classes/chokolate_milk", "model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/chomilk.hdf5")
#del trained_model
#print("starting with yogurt")
#trained_model = train(model(), "Dataset/seperated_food_images/extraced_classes/yoghurt", "model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/yogurt.hdf5")
##del trained_model
#print("starting with milk")
#trained_model = train(model(), "Dataset/seperated_food_images/extraced_classes/milk", "model_data/navn_vekt_fasit_2_0")
#trained_model.save("model_data/trained_weights_/milk.hdf5")
#del trained_model

#print(xl["Navn"][1])
#print(xl)
#img = Image.open("Dataset/Bilder/Brød_1v_05.341.jpg")
#img = blackout(img, 1000, 1000, 1200, 1000)
#img.show()
#import time
#start = time.time()
#blackout("Dataset/Bilder (1)/xml/evaluation/Sjokolademelk_13_01.008.jpg", 1564,3470,2037,3865)
#loop_files("Dataset/Bilder>300", "Dataset/fixed_images/")

#print("time taken: ", time.time() - start)







