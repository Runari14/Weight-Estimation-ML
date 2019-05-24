import xml.etree.ElementTree as ET
import os

#sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["weight", "post-it"]

image_data_path = "Dataset/Bilder/xml"

def rename_images(path:str):
	"""
	rename the images to have more clean names.
	eg: no spaces and no ','
	:param path:
	:return:
	"""
	files = os.listdir(path)

	for file in files:
		filepath = path + "/" + file

		name = file.replace(" ", "_")
		name = name.replace(",", "")
		name = name.replace("æ", "ae'")
		name = name.replace("ø", "oe'")
		name = name.replace("å", "aa'")
		name = name.replace("ü", "u")
		new_filepath = path + "/" + name
		os.rename(filepath, new_filepath)



def convert_xml(xml_path, out_file):
	in_file = open(xml_path,encoding="utf-8")
	tree = ET.parse(in_file)
	root = tree.getroot()

	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		class_name = obj.find('name').text
		if class_name not in classes or int(difficult) == 1:
			continue
		class_id = classes.index(class_name)
		xmlbox = obj.find('bndbox')
		cords = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
			 int(xmlbox.find('ymax').text))
		out_file.write(",".join([str(a) for a in cords]) + ',' + str(class_id))


def get_xml_data(save_location:str):
	for group in ["evaluation", "test", "training"]:
		group_path = image_data_path + "/" + group
		files = os.listdir(group_path)

		image_ids = [image_id.replace(".xml", "").replace(".jpg", "") for image_id in files]
		out_file = open("{}/{}.txt".format(save_location, group), "w+", encoding="utf-8")

		for image_id in image_ids:
			image_path = group_path + "/" + image_id
			out_file.write(image_path + ".jpg,")

			convert_xml(image_path + ".xml", out_file)

			out_file.write("\n")

		out_file.close()

def write_classes(save_loaction_path:str, classes:list):
	f = open(save_loaction_path, "w+", encoding="utf-8")
	for c in classes:
		f.write(c + "\n")
	f.close()


rename_images("Dataset/Bilder>300")
#get_xml_data("model_data/")
#write_classes("model_data/post_it_classes.txt", classes)

