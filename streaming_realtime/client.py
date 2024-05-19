import cv2
import zmq,time
from threading import Thread
#Hyper parameter
server_ip = "192.168.1.186"

cap = cv2.VideoCapture(0)
class SenderClient:
	"""
		Sender will connect to particular tcp destination, if it's listening
		Call "send(image)" to send image to server, image could be collected from cv2
	"""
	def __init__(self,dest_host=server_ip,port=5555):
		self.dest_host = dest_host
		self.context = zmq.Context()
		self.socket = self.context.socket(zmq.PUSH)
		self.socket.connect(f"tcp://{dest_host}:{port}")
	def send(self,image):
		_, buffer = cv2.imencode('.jpg', image)
		image_bytes = buffer.tobytes()
		self.socket.send(image_bytes)
	def __del__(self):
		self.socket.close()


class ReceiverClient(Thread):
	"""
		Receiver run as a thread in program, which is listening reply label from Server
		Just init and call "start"
		If needed, get "sentence" from "self.sentence"
	"""
	def __init__(self,host=server_ip,port=5556):
		super(ReceiverClient,self).__init__()
		self.host = host
		self.port = port
		self.sentence = []
		self.stop_flag = False
	def run(self):
		context = zmq.Context()
		socket = context.socket(zmq.PULL)
		socket.connect(f"tcp://{self.host}:{self.port}")
		print("Success")
		while not self.stop_flag:
			try:
				message = socket.recv_string(flags=zmq.NOBLOCK)
				self.sentence.append(message)
				self.sentence = self.sentence[-5:]
				print(f"Server: {message}")
			except:
				time.sleep(0.1)
		socket.close()
		context.term()

receiver = ReceiverClient()
receiver.start()
sender = SenderClient()
try:
	while cap.isOpened():
		ret,frame = cap.read()
		sender.send(frame)
		cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
		cv2.putText(frame, ' '.join(receiver.sentence), (3,30), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.imshow("Fullscreen",frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

except:
	print("Paused")
finally:
	receiver.stop_flag = True
	receiver.join()
	cv2.destroyAllWindows()
