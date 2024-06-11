import cv2
import zmq,time,pickle
from threading import Thread
#Hyper parameter
server_ip = "192.168.1.186"


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
	def send(self,image,flag = False):
		_, buffer = cv2.imencode('.jpg', image)
		image_bytes = buffer.tobytes()
		mes = pickle.dumps((image_bytes,flag))
		self.socket.send(mes)
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
		self.message = None
		self.stop_flag = False
	def run(self):
		context = zmq.Context()
		socket = context.socket(zmq.PULL)
		socket.connect(f"tcp://{self.host}:{self.port}")
		print("Success")
		while not self.stop_flag:
			try:
				self.message = socket.recv_string(flags=zmq.NOBLOCK)
				print(f"Server: {self.message}")
			except:
				time.sleep(0.1)
		socket.close()
		context.term()

receiver = ReceiverClient()
receiver.start()
sender = SenderClient()
start_flag = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
cap.set(cv2.CAP_PROP_FPS, 15)

from gpiozero import Button
button = Button(21)
cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
print("Waiting for start...")
img = cv2.imread("Picture1.jpg")
cv2.imshow("FullScreen", img)
cv2.waitKey(100)
try:
	while cap.isOpened():
		
		ret,frame = cap.read()
		if receiver.message is not None:
			cv2.putText(frame, f'{receiver.message}', (80,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255, 0), 6, cv2.LINE_AA)
			cv2.imshow("FullScreen", frame)
			receiver.message = None
			start_flag = True
			if cv2.waitKey(2000) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			while not button.is_pressed:
				cv2.imshow("FullScreen", img)	
				cv2.waitKey(1000) 
		else:
			sender.send(frame,start_flag)
			if start_flag == True:
				start_flag = False
			cv2.imshow("FullScreen",frame)
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

except Exception as e:
	print(e)
	print("Paused")
finally:
	receiver.stop_flag = True
	receiver.join()
	cv2.destroyAllWindows()
