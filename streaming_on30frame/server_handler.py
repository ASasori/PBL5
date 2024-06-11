import cv2,sys,os,json,numpy as np,copy
from time import sleep
import mediapipe as mp
sys.path.append(os.path.abspath("./"))
from models.model_train_13.classes import load_model


class ModelSolver:
    def __init__(self,threshold = 0.8,num_frame = 30,imgsize = (640,480)):
        #hyper parameter
        self.width,self.height = imgsize
        #model and mediapipe
        self.model = load_model()
        self.mp_holistic = mp.solutions.holistic 
        self.mp_drawing = mp.solutions.drawing_utils
        #global parameter
        self.last = None
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = threshold
        self.num_frame = num_frame
        with open("label_list.json") as js:
            self.actions = list(json.load(js).values())
        print(self.actions)
        self.delay = 0
        self.trus = 0
        self.count = -1
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("Initialized ModelSolver")

    def mediapipe_detection(self,image, model):
        # từ image, model dự đoán trả về kết quả (định dạng mặc định) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    def draw_landmarks(self,image, results):
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def update_mpresult(self,res,results,last):
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
    
    def normalize_keypoint(self,res,img=None):
        #normalize keypoint
        x1,y1,x2,y2 = res[11][0]*self.width,res[11][1]*self.height,res[12][0]*self.width,res[12][1]*self.height
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
        scale = (200*self.width/640)/dis
        for i in range(len(res)):
            if res[i][0]==0 and res[i][1]==0:
                continue
            res[i][0] = vector[0]+res[i][0]
            res[i][1] = vector[1]+res[i][1]
            res[i][0] = 0.5+(res[i][0]-0.5)*scale
            res[i][1] = 0.5+(res[i][1]-0.5)*scale
        return res

    def extract_keypoint(self,results,last):
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

    def extract_keypoints_flatten(self,result,last = None,img = None):
        #đây là hàm chính thức
        res = self.extract_keypoint(result,last)
        res = self.normalize_keypoint(res,img)
        self.update_mpresult(res,result,last)
        return np.concatenate([x for x in res])
    
    def solve(self,image):
        # append image, return None or label
        self.count+=1
        frame, results = self.mediapipe_detection(image, self.holistic)
        keypoints = self.extract_keypoints_flatten(results,self.last)
        self.last = copy.deepcopy(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.num_frame:]
        ret = None
        if self.delay!=0:
            self.delay-=1
        elif self.delay ==0 and len(self.sequence)>=self.num_frame and self.count%5==0:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))
            print(f"Predicted {self.actions[np.argmax(res)]}")
            self.predictions = self.predictions[-5:]
            if np.unique(self.predictions)[0]==np.argmax(res):
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence)>0:
                        if self.actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(self.actions[np.argmax(res)])
                            ret = self.sentence[-1]
                            self.delay = 30
                    else:
                        self.sentence.append(self.actions[np.argmax(res)])
                        ret = self.sentence[-1]
                        self.delay =30
            if len(self.sentence) > 2: 
                self.sentence = self.sentence[-2:]
        return ret
    
    def solve_on_30_frames(self,images):
        sequence = []
        self.last = None
        for image in images:
            frame, results = self.mediapipe_detection(image, self.holistic)
            keypoints = self.extract_keypoints_flatten(results,self.last)
            self.draw_landmarks(image=image,results=results)
            self.last = copy.deepcopy(results)
            sequence.append(keypoints)
            cv2.imshow("FullScreen", image)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        sequence = np.array(sequence)
        sequence = np.hstack((sequence[:,:92], sequence[:,132:]))
        res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        return self.actions[np.argmax(res)]

    def __del__(self):
        pass
