from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
# initialize the ImageHub object
imageHub = imagezmq.ImageHub()

while True:
	(rpiName, frame) = imageHub.recv_image()
	print(rpiName,frame.shape)
	cv2.imshow("Test_stream",frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	imageHub.send_reply(b'OK')