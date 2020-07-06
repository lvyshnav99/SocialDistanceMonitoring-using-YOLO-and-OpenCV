##necessary packages
from google.colab.patches import cv2_imshow
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import sys

labelspath=os.path.sep.join(["/content/social-distance-detector/yolo-coco/coco.names"])
labels=open(labelspath).read().strip().split("\n")

personlabel=labels.index('person')

weigthpath=os.path.sep.join(["/content/social-distance-detector/yolo-coco/yolov3.weights"])
cnfgpath=os.path.sep.join(["/content/social-distance-detector/yolo-coco/yolov3.cfg"])

net=cv2.dnn.readNetFromDarknet(cnfgpath,weigthpath)

layernames=net.getLayerNames()
print(layernames)


ln=[layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]

frame=mimage.imread("/content/1.jpg")
frame=imutils.resize(frame,width=700,height=500)
print(frame.shape)
result=detect_people(frame,net,ln,personlabel)

print(len(result))

if len(result)<2:
  print("People are perfectly maintaining social distancing")
  plt.imshow(frame)
  sys.exit(...)


centroids=[r[2] for r in result]

##finding distance between each pair of centroids

D=dist.cdist(centroids,centroids,metric="euclidean")

voilatedlist=set()

for i in range(0,D.shape[0]):
  for j in range(i+1,D.shape[1]):
    if D[i,j]<MIN_DISTANCE:
      voilatedlist.add(i)
      voilatedlist.add(j)
  


for (i,(confidence,box,centroids)) in enumerate(result):
  startx,starty,endx,endy=box
  centerx,centery=centroids
  color=(0,255,0)
  if i in voilatedlist:
    color=(255,0,0)
  cv2.rectangle(frame,(startx,starty),(endx,endy),color,5)
  cv2.circle(frame,(centerx,centery),5,(0,0,255),2)
plt.imshow(frame)