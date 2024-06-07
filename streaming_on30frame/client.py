import cv2
import zmq,time
from threading import Thread
import numpy as np
import pickle
#Hyper parameter
server_ip = "192.168.1.186"

cap = cv2.VideoCapture(0)

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

try:
	while cap.isOpened():
		imgs = []
		for i in range(30):
			if (not cap.isOpened()): break
			ret,frame = cap.read()
			imgs.append(frame)
			cv2.imshow("Fullscreen",frame)
			if cv2.waitKey(50) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
		sender.send(imgs)
		label = sender.recv()
		# Print result
		image = imgs[-1]
		cv2.putText(image, f'Predicted: {label}', (80,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
		cv2.imshow("Fullscreen", image)
		print(label)
		if cv2.waitKey(2000) & 0xFF == ord('q'):
			break

except Exception as e:
	print(e)
	print("Paused")
