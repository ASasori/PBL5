import cv2
import requests
import base64
from time import time
cap = cv2.VideoCapture('5.mp4')
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    frames.append(encoded_image)
cap.release()

if frames:
    payload = {
        "images": frames  # You can send multiple images by adding more encoded strings to the list
    }

    # Define the API endpoint
    # api_url = "http://127.0.0.1:5000/predict"
    api_url = "http://bird-faithful-hagfish.ngrok-free.app/predict"
    # Send the request
    start = time()
    response = requests.post(api_url, json=payload)
    # Check for successful response
    if response.status_code == 200:
        results = response.json()
        print("Received results:", results)
    else:
        print("Error:", response.status_code, response.text)
    end = time()
    print('Response test time on localhost:',end-start)
