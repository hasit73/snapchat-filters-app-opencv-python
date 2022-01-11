import cv2
import mediapipe as mp
import numpy as np
from time import time
from scipy.spatial.distance import euclidean

class HandDetector:
    def __init__(self,static_image_mode=False,max_num_hands = 2,min_detection_confidence = 0.6,min_tracking_confidence = 0.6):
        self.__mpHand = mp.solutions.hands
        self.__detector = self.__mpHand.Hands(static_image_mode=static_image_mode,
                                                max_num_hands = max_num_hands,
                                                min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
        self.__mpDraw = mp.solutions.drawing_utils

        self.__one_hand_distance_pairs = np.array([[4,9],[5,8],[9,12],[13,16],[17,20]])
        self.__hand_landmarks = []


        

    def find_hands(self,image):
        results = self.__detector.process(image)
        # counter = 1
        all_points = []
        self.__hand_landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w) , int(lm.y*h)
                    all_points.append([cx,cy])                    
                self.__hand_landmarks.append(handLms)
        return all_points
    


    def draw_fingers(self,image,all_points):
        num_fingers = 0
        
        fingers_up_flag = []
        for handLms in self.__hand_landmarks:
            self.__mpDraw.draw_landmarks(image,handLms,self.__mpHand.HAND_CONNECTIONS)
        if(len(all_points)):
            rect_points = np.array([all_points[0],all_points[1],all_points[5],all_points[17]])
            rect_points = rect_points.reshape((-1,1,2))
            
            for i1,i2 in self.__one_hand_distance_pairs:
                if(euclidean(all_points[i1],all_points[i2]) >50):
                    num_fingers+=1
                    fingers_up_flag.append(1)
                else:
                    fingers_up_flag.append(0)
            if(len(all_points)>21):
                for i1,i2 in (21+self.__one_hand_distance_pairs):
                    if(euclidean(all_points[i1],all_points[i2]) >50):
                        num_fingers+=1
                        fingers_up_flag.append(1)
                    else:
                        fingers_up_flag.append(0)
        cv2.rectangle(image,(3,3),(180,40),(255,255,255),-1)
        cv2.putText(image,f"Fingers : {num_fingers}",(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (145,0 , 174), 2)
        return image