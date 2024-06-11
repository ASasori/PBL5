import cv2
import zmq,time,sys,os,pickle
import numpy as np
from threading import Thread,Event
sys.path.append(os.path.abspath("./"))
from streaming_realtime.server_handler import ModelSolver
class ServerReceiver:
    """
        Run with threading, always collect data from client
        But only process with the last frame.
    """
    def __init__(self,port=5555):
        super(ServerReceiver,self).__init__()
        #initialize data for class
        self.port = port
        self._stop = False
        self.image = None
        self.start = False
        #collector socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{port}")
        #event handler and threading.
        self.data_ready = Event()
        self._thread = Thread(target=self._run,args=())
        self.daemon = True
        self._thread.start()
    def _run(self):
        # a loop always waiting for 
        while not self._stop:
            mes = self.socket.recv()
            image_bytes,flag = pickle.loads(mes)
            if flag:
                self.start = True
                continue
            np_arr = np.frombuffer(image_bytes,np.uint8)
            self.image = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
            self.data_ready.set()
    def recvImg(self):
        if not self.start:
            return None
        flag = self.data_ready.wait()
        self.data_ready.clear()
        return self.image
    def close(self):
        self._stop = True

class ServerMsgSender:
    """
        Just for send label predicted to client
    """
    def __init__(self,port=5556):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(f"tcp://*:{port}")#port 5556 giao tiếp message
    def send(self,msg):
        self.socket.send_string(msg)
        
receiver = ServerReceiver()
sender = ServerMsgSender()
solver = ModelSolver()
while True:
    image = receiver.recvImg()
    if image is None:
        continue
    label = solver.solve(image)
    if label:
        print("Sending label: ",label)
        sender.send(label)
        receiver.start = False

