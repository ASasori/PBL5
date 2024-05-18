from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

server_ip = "192.168.1.186"
sender = imagezmq.ImageSender(connect_to=f"tcp://{server_ip}:5555")
rpiName = socket.gethostname()
vs = VideoStream().start()
#vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
 
while True:
	# read the frame from the camera and send it to the server
	frame = vs.read()
	sender.send_image(rpiName, frame)