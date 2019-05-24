import json
from pprint import pprint
import os
from PIL import Image

import numpy as np

# file = open("Dataset/seperated_food_images/Appelsinjuice_1_06.568.json")
# j = json.load(file)
# file.close()
# pprint(j[0]["label"])

# amounts = {}
classes = ["milk", "bread", "yoghurt", "crispbread", "chokolate_milk", "cheese"]


def extract(img, x_min :int, y_min :int, x_max :int, y_max :int):
	img_array = np.array(img)
	img_array = img_array[y_min:y_max, x_min:x_max]
	img = Image.fromarray(img_array)
	#img.show()
	return img

for file_path in os.listdir("Dataset/seperated_food_images"):
	# check for potential faults
	if not file_path.endswith(".json"):
		continue
	path = "Dataset/seperated_food_images/" +file_path

	if os.path.isdir(path):
		continue


	file = open(path)
	js = json.load(file)
	labels = []
	for c in js:
		label = c["label"]
		# if label == "bun":
		#	print(path)
		if label in ["weight", "post-it"]:
			continue

		if label in classes: # only use files with more than 10 members, that is not post-it or weight

			print \
				("x_min:{}, x_max{}, y_min{}, y_max{}".format(c["topleft"]["x"], c["bottomright"]["x"], c["topleft"]["y"]
				                                            ,c["bottomright"]["y"]))
			img_adr = path.replace("json" ,"jpg")
			img = Image.open(img_adr)
			#img.show()
			x_min, x_max, y_min, y_max = c["topleft"]["x"], c["bottomright"]["x"], c["topleft"]["y"] ,c["bottomright"]["y"]

			# expand the image a bit more than what the box says, so that we can get more information, and weird cropping hopefully won't happen.
			# mins
			if x_min - 100 < 0:
				x_min = 0
			else:
				x_min -= 100

			if y_min - 100 < 0:
				y_min = 0
			else:
				y_min -= 100

			# max
			width, height = img.size
			if x_max + 100 < width:
				x_max = width
			else:
				x_max += 100

			if y_max + 100 < height:
				y_max = height
			else:
				y_max += 100



			img = extract(img ,x_min ,y_min ,x_max ,y_max)
			img.save("Dataset/seperated_food_images/extraced_classes/{}/{}".format(label, file_path.replace(".json", ".jpg")))
			#img.show()

		"""if label in amounts:
			amounts[label] +=1
		else:
			amounts[label] = 1
		"""

	# print(label)
	# if label in labels:
	#	print("repeat of {} in {}".format(label, file))
	# labels.append(label)
	# print("--\n")

	file.close()

"""
count = 0
for key in amounts:
	if amounts[key] >= 10:
		os.mkdir("Dataset/seperated_food_images/extraced_classes/"+key)

		#print(key, amounts[key])
		print(amounts[key])
		count += 1

print("of {} classes, {} are more than 10".format(len(amounts), count))
#pprint(amounts)
"""

