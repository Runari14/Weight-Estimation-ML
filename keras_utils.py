import keras


def load(path: str):
	return keras.models.load_model(path)


def freeze(model, layer_start):
	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 249 layers and unfreeze the rest:

	for layer in model.layers[:layer_start]:  # freeze layers before start
		layer.trainable = False
	for layer in model.layers[layer_start:]:  # unfreeze layers after start
		layer.trainable = True


def decode(l, dataset_path, top=3):
	l = list(l[0])
	import os
	# find the best
	answers = []
	for x in range(top):
		value = max(l)
		index = l.index(value)
		l.remove(value)
		answers.append((index, value))

	# find the labels
	names = []
	for name in os.listdir(dataset_path):
		names.append(name)
	names = sorted(names)

	# insert the labels

	labeled = []
	for a in answers:
		label = names[a[0]]
		labeled.append((label, a[1]))
	return labeled

def æøå(path):
	"""
	takes in a name and translates it so that æøå is correct
	:param name:
	:return:
	"""
	import os
	for file in os.listdir(path):
		file2 = file.replace("├╕", "ø")
		file2 = file2.replace("├е", "aa")
		file2 = file2.replace("├ж", "ae")

		file2 = file2.replace("ø", "ø")
		file2 = file2.replace("å", "aa")
		file2 = file2.replace("æ", "ae")
		file2 = file2.replace("'", "")

		path_old = path+"/"+file
		path_new = path+"/"+file2
		print(path_old)
		os.rename(path_old, path_new)
	return


æøå("../Dataset/fixed_images (copy)")


