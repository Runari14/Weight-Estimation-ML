from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions, InceptionResNetV2
import numpy as np

resnet = InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


################################### TEST IMAGES ###################################
test_image1 = "Dataset/Food-101/images/cup_cakes/15425.jpg"
test_image2 = "Dataset/food-101/images/baby_back_ribs/2432.jpg"
test_image3 = "Dataset/Food-101/images/pizza/32004.jpg"
test_image4 = "Dataset/UECFood100/1/1.jpg"

img_path = test_image2

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

preds = resnet.predict(x)

print(decode_predictions(preds))
