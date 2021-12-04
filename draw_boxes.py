from time import sleep

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

coco = COCO('data/annotations/000000_coco.json')
# print(coco)
ids = coco.imgs.keys()
# print(ids)
anns = coco.loadAnns(ids)
# print(anns[0])
[x, y, w, h] = anns[0]['bbox']
[x2, y2, w2, h2] = anns[1]['bbox']
img = cv2.imread('data/ambient/000000_ambient/frame_000000.PNG')

# sleep(100)
cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 5)
cv2.rectangle(img, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), (255, 0, 0), 5)
cv2.imshow('', img)
cv2.waitKey(0)
# sleep(100)
# for ann in anns:
#     print(ann)
#     [x, y, w, h] = ann['bbox']

