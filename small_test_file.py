import os
import shutil
path = "Dataset/seperated_food_images/classes"
source = "Dataset/fixed_images (copy)/"
for dir in os.listdir(path):
	for image in os.listdir(path + "/" + dir):
		shutil.copy(source+image, path + "/" + dir + "/" + image)








