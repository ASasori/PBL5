from imutils.video import VideoStream
from threading import Thread
import imagezmq
import cv2
import socket
import time
import zmq

server_ip = "192.168.1.186"
sender = imagezmq.ImageSender(connect_to=f"tcp://{server_ip}:5555")
rpiName = socket.gethostname()
cap = cv2.VideoCapture(0)
stop_flag = False
print("Starting collect image...")
time.sleep(1.0)
sentence = []
class Receiver(Thread):
	def __init__(self,host=server_ip,port=5556):
		self.host = host
		self.port = port
		super(Receiver,self).__init__()
	def run(self):
		global sentence
		context = zmq.Context()
		socket = context.socket(zmq.PULL)
		socket.connect(f"tcp://{self.host}:{self.port}")
		print("Success")
		while not stop_flag:
			try:
				message = socket.recv_string(flags=zmq.NOBLOCK)
				sentence.append(message)
				sentence = sentence[-5:]
				print(f"Server: {message}")
			except:
				time.sleep(0.1)
		socket.close()
		context.term()
receiver = Receiver()
receiver.start()
try:
	while cap.isOpened():
		ret,frame = cap.read()
		sender.send_image(rpiName, frame)
		cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
		cv2.putText(frame, ' '.join(sentence), (3,30), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.imshow("Fullscreen",frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
except KeyboardInterrupt:
	print("Stop by break")
finally:
	stop_flag = True
	receiver.join()
	cv2.destroyAllWindows()
	sender.close()