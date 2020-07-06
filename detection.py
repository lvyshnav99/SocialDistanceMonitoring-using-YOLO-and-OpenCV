import cv2
import numpy as np
import matplotlib.image as mimage
import matplotlib.pyplot as plt

def detect_people(frame,net,ln,personlabel):
  
  (H,W)=frame.shape[0],frame.shape[1]
  results=[]

  blob=cv2.dnn.blobFromImage(frame,1/255.0,size=(416,416),swapRB=True,crop=False)

  net.setInput(blob)
  outputs=net.forward(ln)

  ###
  boxes=[]
  confidences=[]
  centroids=[]
  
  for output in outputs:
    for detection in output:
      scores = detection[5:]
      classid=np.argmax(scores)
      confidencescore=scores[classid]

      if classid==personlabel and confidencescore>MIN_CONF:
        box=detection[0:4]*np.array([W,H,W,H])
        (centerX, centerY, width, height) = box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        boxes.append([x, y, int(width), int(height)])
        centroids.append((centerX, centerY))
        confidences.append(float(confidencescore))
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

  if len(idxs)>0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      r = (confidences[i], (x, y, x + w, y + h), centroids[i])
      results.append(r)
  return results