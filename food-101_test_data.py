import random
import os
import csv
from functools import reduce
from shutil import move, copy
random.seed(3)

data_structures = ["food11", "food101"]
d = {'apple_pie': 'Dessert', 'baby_back_ribs': 'Meat', 'baklava': 'Dessert', 'beef_carpaccio': 'Meat', 'beef_tartare': 'Meat', 'beet_salad': 'Vegetable&Fruit', 'beignets': 'Dessert', 'bibimbap': 'Egg', 'bread_pudding': 'Dessert', 'breakfast_burrito': 'Vegetable&Fruit', 'bruschetta': 'Bread', 'caesar_salad': 'Vegetable&Fruit', 'cannoli': 'Dessert', 'caprese_salad': 'Vegetable&Fruit', 'carrot_cake': 'Dessert', 'ceviche': 'Seafood', 'cheesecake': 'Dessert', 'cheese_plate': 'Dairy products', 'chicken_curry': 'Meat', 'chicken_quesadilla': 'Meat', 'chicken_wings': 'Meat', 'chocolate_cake': 'Dessert', 'chocolate_mousse': 'Dessert', 'churros': 'Dessert', 'clam_chowder': 'Seafood', 'club_sandwich': 'Bread', 'crab_cakes': 'Seafood', 'creme_brulee': 'Dessert', 'croque_madame': 'Egg', 'cup_cakes': 'Dessert', 'deviled_eggs': 'Egg', 'donuts': 'Dessert', 'dumplings': 'Dessert', 'edamame': 'Vegetable&Fruit', 'eggs_benedict': 'Egg', 'escargots': 'Seafood', 'falafel': 'Vegetable&Fruit', 'filet_mignon': 'Meat', 'fish_and_chips': 'Seafood', 'foie_gras': 'Meat', 'french_fries': 'Fried food', 'french_onion_soup': 'Soup', 'french_toast': 'Bread', 'fried_calamari': 'Fried food', 'fried_rice': 'Rice', 'frozen_yogurt': 'Diary product', 'garlic_bread': 'Bread', 'gnocchi': 'Noodles&Pasta', 'greek_salad': 'Vegetable&Fruit', 'grilled_cheese_sandwich': 'Bread', 'grilled_salmon': 'Seafood', 'guacamole': 'Vegetable&Fruit', 'gyoza': 'Meat', 'hamburger': 'Meat', 'hot_and_sour_soup': 'Soup', 'hot_dog': 'Meat', 'huevos_rancheros': 'Vegetable&Fruit', 'hummus': 'Vegetable&Fruit', 'ice_cream': 'Dessert', 'lasagna': 'Noodles&Pasta', 'lobster_bisque': 'Soup', 'lobster_roll_sandwich': 'Bread', 'macaroni_and_cheese': 'Noodles&Pasta', 'macarons': 'Dessert', 'miso_soup': 'Soup', 'mussels': 'Seafood', 'nachos': 'Vegetable&Fruit', 'omelette': 'Egg', 'onion_rings': 'Fried food', 'oysters': 'Seafood', 'pad_thai': 'Noodles&Pasta', 'paella': 'Seafood', 'pancakes': 'Dessert', 'panna_cotta': 'Dessert', 'peking_duck': 'Meat', 'pho': 'Soup', 'pizza': 'Vegetable&Fruit', 'pork_chop': 'Meat', 'poutine': 'Fried food', 'prime_rib': 'Meat', 'pulled_pork_sandwich': 'Meat', 'ramen': 'Noodles&Pasta', 'ravioli': 'Noodles&Pasta', 'red_velvet_cake': 'Dessert', 'risotto': 'Rice', 'samosa': 'Fried food', 'sashimi': 'Seafood', 'scallops': 'Seafood', 'seaweed_salad': 'Vegetable&Fruit', 'shrimp_and_grits': 'Seafood', 'spaghetti_bolognese': 'Noodles&Pasta', 'spaghetti_carbonara': 'Noodles&Pasta', 'spring_rolls': 'Vegetable&Fruit', 'steak': 'Meat', 'strawberry_shortcake': 'Dessert', 'sushi': 'Seafood', 'tacos': 'Vegetable&Fruit', 'takoyaki': 'Dessert', 'tiramisu': 'Dessert', 'tuna_tartare': 'Seafood', 'waffles': 'Dessert'}


def select_images(path:str, dest:str, data_structure):
	if not os.path.isdir(dest+"/"+data_structures[0]):
		os.mkdir(dest+"/"+data_structures[0])

	if not os.path.isdir(dest+"/"+data_structures[1]):
		os.mkdir(dest+"/"+data_structures[1])


	for clas in os.listdir(path):
		old_dir = path +"/"+ clas
		images = os.listdir(old_dir)
		random.shuffle(images)
		images = images[:10]
		new_101_class = clas

		new_11_class = d[clas]

		new_101_dir = dest + "/"+data_structures[1]+ "/" + new_101_class
		new_11_dir = dest+"/"+data_structures[0]+"/"+new_11_class
		if not os.path.isdir(new_101_dir):
			os.mkdir(new_101_dir)
		if not os.path.isdir(new_11_dir):
			os.mkdir(new_11_dir)

		for image in images:
			copy(old_dir + "/" + image, new_101_dir+"/"+image)
			move(old_dir + "/" + image, new_11_dir+"/"+image)

select_images("Dataset/food-101/images", "Dataset/food-101/Test_data", "11")


