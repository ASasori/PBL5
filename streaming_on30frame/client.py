import cv2
import zmq,time
from threading import Thread
import numpy as np
import pickle
#Hyper parameter
server_ip = "192.168.0.106"

cap = cv2.VideoCapture(1)

class Sender:
	def __init__(self,server_ip = server_ip):
		self.server_ip = server_ip
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.REQ)
		self.socket.connect(f"tcp://{self.server_ip}:5555")
	def send(self,frames):
		images = self.load_image(frames)
		message = pickle.dumps(images)
		self.socket.send(message)
	def recv(self):
		return self.socket.recv_string()
	def load_image(self,images):
		res = []
		for frame in images:
			_,buffer = cv2.imencode(".jpg",frame)
			res.append(buffer.tobytes())
		return res
sender = Sender(server_ip)
# from gpiozero import Button
# button = Button(21)
cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("Waiting for start...")
# cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
img = cv2.imread("Picture1.jpg")
# img = cv2.resize(cv2.imread("Picture1.jpg"),(480,320))
cv2.imshow("FullScreen", img)
cv2.waitKey(100)
# button.wait_for_press()
cv2.CV_CAP_PROP_POS_FRAMES
try:
	while cap.isOpened():
		# if not button.is_pressed:
		# 	cv2.imshow("FullScreen", img)	
		# 	cv2.waitKey(1000) 
		# 	continue
		imgs = []
		for i in range(30):
			if (not cap.isOpened()): break
			ret,frame = cap.read()
			imgs.append(frame)
			cv2.imshow("FullScreen",frame)
			if cv2.waitKey(80) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
		sender.send(imgs)
		label = sender.recv()
		image = imgs[-1]
		cv2.putText(image, f'{label}', (80,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255, 0), 6, cv2.LINE_AA)
		cv2.imshow("FullScreen", image)
		print(label)
		if cv2.waitKey(2000) & 0xFF == ord('q'):
			break

except Exception as e:
	print(e)
	print("Paused")
