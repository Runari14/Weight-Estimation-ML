from darkflow.net.build import TFNet

options = {
    "model": "cfg/tiny-yolo-voc-59c.cfg",
    "load": "bin/tiny-yolo-voc.weights",
    "batch": 32,
    "epoch": 500,
    "train": True,
    "annotation": "F:/Dataset/FoodX/labels",
    "dataset": "F:/Dataset/FoodX/images",
    "gpu": 1.0
}

tfnet = TFNet(options)

tfnet.train()