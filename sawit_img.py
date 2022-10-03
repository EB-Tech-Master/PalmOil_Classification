from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os, json, cv2, random
from matplotlib import pyplot as plt
import numpy as np


classname = {
  "0": "overripe",
  "1": "ripe",
  "2": "underripe",
  "3": "unripe",
  
}


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # Set threshold for this model
cfg.MODEL.WEIGHTS = 'C:/Users/User/Documents/EBTECH/EB_Detectron2/model_final_101.pth' # Set path model .pth
cfg.MODEL.DEVICE = "cpu" # cpu or cuda
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
predictor = DefaultPredictor(cfg)

im = cv2.imread("C:/Users/User/Desktop/more.PNG")
im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
overlay = im.copy()
outputs = predictor(im)

mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
num_instances = mask_array.shape[0]
scores = outputs['instances'].scores.to("cpu").numpy()
labels = outputs['instances'].pred_classes .to("cpu").numpy()
bbox   = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()
print(labels)
mask_array = np.moveaxis(mask_array, 0, -1)

mask_array_instance = []
point_polygon_instance =[]
show = [str(j)+" : "+classname[str(j)]for j in labels]
print(show)
h = im.shape[0]
w = im.shape[1]
img_mask = np.zeros([h, w, 3], np.uint8)
for i in range(num_instances):
    name = '{}: {}%'.format(classname[str(labels[i])],int(scores[i]*100))
    color = list(np.random.random(size=3) * 256)
    img = np.zeros_like(im)
    mask_array_instance.append(mask_array[:, :, i:(i+1)])
    img = np.where(mask_array_instance[i] == True, 255, img)
    array_img = np.asarray(img)
    img_mask2 = cv2.cvtColor(array_img, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(img_mask2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im2, contours, -1, (0,255,0), 3)
    cv2.rectangle(im2, (round(bbox[i][0]), round(bbox[i][1])), (round(bbox[i][2]), round(bbox[i][3])), color, 3)
    cv2.rectangle(im2, (round(bbox[i][0]), round(bbox[i][1])), (int(round(bbox[i][0]))+100, int(round(bbox[i][1]))+30), (0, 0, 0), cv2.FILLED)
    cv2.putText(im2, name, (int(bbox[i][0]), int(bbox[i][1])+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    cv2.polylines(im2, contours, True, color, 3)
    cv2.fillPoly(overlay, contours, color,)
    image_new = cv2.addWeighted(overlay, 0.4, im2, 1 - 0.4, 0)

cv2.imshow("image", image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()