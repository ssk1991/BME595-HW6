# Adapted from transfer learning tutorial

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.image as img
from PIL import Image
import numpy as np
import cv2
import time

model = torch.load("save_model.pt")


# ===== Put text =========
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    resized_image = cv2.resize(frame, (224, 224))
    cv2.imwrite('frame.png', resized_image)
    input_image = img.imread("frame.png")
    input_image = torch.from_numpy(input_image)
    input_image = Variable(input_image)
    input_image = input_image.view(1, 3, 224, 224)
    outputs = model(input_image)
    _, pred = torch.max(outputs.data, 1)
    # pred = pred.data.numpy()
    # pred_label = np.argmax(pred, axis=1)
    cv2.putText(frame, str(pred),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
