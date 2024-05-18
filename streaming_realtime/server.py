"""pub_sub_receive.py -- receive OpenCV stream using PUB SUB."""

import zmq,cv2,sys,os,json,numpy as np,copy,csv
from time import sleep
import mediapipe as mp
from server_handler import VideoStreamSubscriber
sys.path.append(os.path.abspath("./"))
from models.model_train_8.classes import load_model

model = load_model()

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

width = 640
height = 480
def mediapipe_detection(image, model):
    # từ image, model dự đoán trả về kết quả (định dạng mặc định) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def update_mpresult(res,results,last):
    c = 0
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark:
            p.x = res[c][0]
            p.y = res[c][1]
            if(c==20 and p.y>1.1 and last): last.right_hand_landmarks = None
            elif(c==19 and p.y>1.1 and last): last.left_hand_landmarks = None
            c+=1
    else:
        for _ in range(33):
            c+=1
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark:
            p.x = res[c][0]
            p.y = res[c][1]
            c+=1
    else:
        if last!=None and last.left_hand_landmarks: results.left_hand_landmarks = copy.deepcopy(last.left_hand_landmarks)
        for _ in range(21):
            c+=1
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark:
            p.x = res[c][0]
            p.y = res[c][1]
            c+=1
    else:
        if last!=None and last.right_hand_landmarks: results.right_hand_landmarks = copy.deepcopy(last.right_hand_landmarks)
        for _ in range(21):
            c+=1
    return results

def normalize_keypoint(res,img=None):
    #normalize keypoint
    x1,y1,x2,y2 = res[11][0]*width,res[11][1]*height,res[12][0]*width,res[12][1]*height
    try:
        cv2.circle(img,(int(x1),int(y1)),4,(0,255,255),-1)
        cv2.circle(img,(int(x2),int(y2)),4,(0,255,255),-1)
    except:
        # print("No img found")
        pass
    dis = np.sqrt((x1-x2)**2+(y1-y2)**2)
    x_cen = (res[11][0]+res[12][0])/2
    y_cen = (res[11][1]+res[12][1])/2
    vector = [0.5-x_cen,0.5-y_cen]
    scale = (200*width/640)/dis
    for i in range(len(res)):
        if res[i][0]==0 and res[i][1]==0:
            continue
        res[i][0] = vector[0]+res[i][0]
        res[i][1] = vector[1]+res[i][1]
        res[i][0] = 0.5+(res[i][0]-0.5)*scale
        res[i][1] = 0.5+(res[i][1]-0.5)*scale
    return res


def extract_keypoint(results,last):
    res = []
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark:
            res.append(np.array([p.x,p.y,p.z,p.visibility]))
    else:
        for _ in range(33):
            res.append(np.array([0,0,0,0]))
    #--------------
    if results.left_hand_landmarks:
        for p in results.left_hand_landmarks.landmark:
            res.append(np.array([p.x,p.y,p.z]))
    elif last!= None and last.left_hand_landmarks:
        for p in last.left_hand_landmarks.landmark:
            res.append(np.array([p.x,p.y,p.z]))
    else:
        for _ in range(21):
            res.append(np.array([0,0,0]))
    #---------------
    if results.right_hand_landmarks:
        for p in results.right_hand_landmarks.landmark:
            res.append(np.array([p.x,p.y,p.z]))
    elif last!=None and last.right_hand_landmarks:
        for p in last.right_hand_landmarks.landmark:
            res.append(np.array([p.x,p.y,p.z]))
    else:
        for _ in range(21):
            res.append(np.array([0,0,0]))
    return res

def extract_keypoints_flatten(result,last = None,img = None):
    #đây là hàm chính thức
    res = extract_keypoint(result,last)
    res = normalize_keypoint(res,img)
    update_mpresult(res,result,last)
    return np.concatenate([x for x in res])

def numpy_to_filecsv(data,filename):
    with open(filename,"w",newline="") as csvfile:
        writer = csv.writer(csvfile,delimiter=",")
        writer.writerows(data.tolist())

def filecsv_to_numpy(filename,data):
    pass
sequence = []
sentence = []
predictions = []
threshold = 0.85
num_frame = 30
with open("label_list.json") as js:
    actions = list(json.load(js).values())
print(actions)
delay = 0
trus = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    count = -1
    #main
    #initialize receiver video stream and message sender
    print("Listening to client...")
    receiver = VideoStreamSubscriber("localhost",5555)
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5556")#port 5556 giao tiếp message
    print("Connected")
    last = None
    while True:
        count+=1
        msg, frame = receiver.receive()
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints_flatten(results,last)
        last = copy.deepcopy(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        image = cv2.flip(image,1)
        if delay !=0:
            delay -=1
        elif delay == 0 and len(sequence)>=num_frame and count%5==0:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            print(res[np.argmax(res)])
            predictions.append(np.argmax(res))

            if np.unique(predictions[-5:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            socket.send_string(actions[np.argmax(res)])
                            delay = 50
                    else:
                        sentence.append(actions[np.argmax(res)])
                        socket.send_string(actions[np.argmax(res)])
                        delay =50
            if len(sentence) > 5: 
                sentence = sentence[-5:]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
receiver.close()