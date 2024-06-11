import cv2
import zmq,time,sys,os
import numpy as np
from threading import Thread,Event
sys.path.append(os.path.abspath("./"))
from streaming_on30frame.server_handler import ModelSolver
import pickle

class Receiver:
    def __init__(self,port = "5555"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
    def recvImgs(self):
        message = self.socket.recv()
        print("Received")
        print(len(message))
        message = pickle.loads(message)
        res = []
        for frame in message:
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            res.append(frame)
        return res
    def send(self,message):
        self.socket.send_string(message)
receiver = Receiver("5555")
solver = ModelSolver()
while True:
    imgs = receiver.recvImgs()
    receiver.send(solver.solve_on_30_frames(imgs))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.destroyAllWindows()
