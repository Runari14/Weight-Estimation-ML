import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    "model": "cfg/tiny-yolo-voc-59c.cfg",
    "load": -1,
    "threshold": 0.1,
    "gpu": 0.7,
    "imgdir": "/home/runar/Documents/master/master_thesis/source/master-era/Dataset/Bilder2",
    "json": True
}

tfnet = TFNet(options)

# img = cv2.imread("F:/Dataset/FoodX/test_images/Broed_1v_05.341.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("F:/Dataset/FoodX/test_images/Broedmaaltid_5f_06.756.jpg", cv2.IMREAD_COLOR)
img = cv2.imread("/home/runar/Documents/UIA/Master/Master Thesis/master-era/Dataset/fixed_images/Broe'd_1a_05.341.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)

from pprint import pprint
pprint(result)

for image in result:
    # print(image["label"]["confidence"])
    tl = (image["topleft"]["x"], image["topleft"]["y"])
    br = (image["bottomright"]["x"], image["bottomright"]["y"])
    label = image["label"]

    img = cv2.rectangle(img, tl, br, (0, 255, 0), 10)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 7)

    plt.imshow(img)
plt.show()
