import cv2
from face_landmark_detection import FaceAnalyzer
from hand_landmark_detection import HandDetector
import numpy as np
from time import time
from itertools import product
class FaceFilters:

    STATIC_FACE_FILTERS = {"red":cv2.imread("./media/glasses_red.png"),
                            "black":cv2.imread("./media/glasses_black.jpg"),
                            "black-frame":cv2.imread("./media/glasses_frame.png"),
                            "cat":cv2.imread("./media/glasses_cat.jpg"),
                            "yellow":cv2.imread("./media/glasses_yellow.jpg"),
                            "modi-mask":cv2.imread("./media/modi-mask.jpg"),
                            "black-mask":cv2.imread("./media/black-mask.jpeg"),
                            "green-mask":cv2.imread("./media/green-mask.jpg"),

                            }
    RESHAPED_FACE_FILTERS = {}

    def __init__(self):
        self.eye_outer_points = []
        self.hand_detector = HandDetector()
        self.face_analyzer = FaceAnalyzer()
        self.__current_filter = "red"
        self.__filters_option_positions = {
                                        "black":[[550,20],[620,70]],
                                        "red":[[550,80],[620,130]],
                                        "black-frame":[[550,140],[620,190]],
                                        "yellow":[[550,200],[620,250]],
                                        "cat":[[550,260],[620,310]],
                                    }
        self.__mask_filters_options = {
                                        "modi-mask":[[50,30],[120,80]],
                                        "black-mask":[[130,30],[200,80]],
                                        "green-mask":[[210,30],[280,80]],
                                    }

        self.__all_filter_positions = {**self.__mask_filters_options,**self.__filters_option_positions}
        
        for k,im in FaceFilters.STATIC_FACE_FILTERS.items():
            FaceFilters.RESHAPED_FACE_FILTERS[k] = cv2.resize(im,(50,30))

    def __draw_home_options(self,image):
        for option_name,pos in self.__all_filter_positions.items():
            cv2.rectangle(image,pos[0],pos[1],(255,255,255),-1)
            image[pos[0][1]+10:pos[0][1]+40,pos[0][0]+10:pos[0][0]+60] = FaceFilters.RESHAPED_FACE_FILTERS[option_name]

    @classmethod
    def set_image(cls,x1,y1,w,h,image,new_glasses,thresh=[220,220,220]):
        combs = product(range(w),range(h))
        for i,j in combs:
            c = new_glasses[j,i]
            val = c>=np.array(thresh)
            if(not np.all(val)):
                im[y1+j,x1+i] = c
        return image
    
    def apply_filter(self,im,filter_name):
        self.face_analyzer.get_face_points(im)
        points = self.face_analyzer.face_points

        if(len(points)):    
            if("mask" in filter_name):
                x1,y1 = points[54][0] - 35,points[54][1] - 57
                x2,y2 = points[365][0] + 75 ,points[365][1]+75
                w = x2-x1-30
                h = y2-y1-30
                if(w!=0 and h!=0):
                    new_img = cv2.resize(FaceFilters.STATIC_FACE_FILTERS[filter_name],(w,h))
                    im = FaceFilters.set_image(x1,y1,w,h,im,new_img,thresh=[190,190,190])
            else:
                x1 = points[130][0]-30
                y1 = points[27][1]-20
                x2 = points[359][0]+30
                y2 = points[253][1]+20
                w = x2-x1
                h = y2-y1
                if(w!=0 and h!=0):
                    new_glasses = cv2.resize(FaceFilters.STATIC_FACE_FILTERS[filter_name],(w,h))
                    im = FaceFilters.set_image(x1,y1,w,h,im,new_glasses)
    @classmethod
    def check_inside_rectangle(cls,rect,p2):
        p2 = np.array(p2)
        # print(p2)
        p1 = rect[0]
        p3 = rect[1]
        if(p1[0] < p2[0] and p2[0] < p3[0] and p1[1] < p2[1] and p2[1] < p3[1]):
            return True
        else:
            return False

    def process_frame(self,image):
        points = self.hand_detector.find_hands(image)
        self.__draw_home_options(image)
        if(len(points)):
            cv2.circle(image,points[8],2,(0,255,0),-1)
            cv2.circle(image,points[12],2,(0,255,255),-1)
            self.__controller(points[8],points[12])
        self.apply_filter(image,self.__current_filter)

    def __controller(self,first_finger,second_finger):
        for option_name,pos in self.__all_filter_positions.items():
            if(FaceFilters.check_inside_rectangle(pos,first_finger) and FaceFilters.check_inside_rectangle(pos,second_finger)):
                self.__current_filter = option_name.lower()

if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,500)
    f1 = FaceFilters()
    ftype = "cat"
    while True:
        st = time()
        ret,im = cap.read()
        im = cv2.flip(im,1)
        f1.process_frame(im)
        et = time()
        cv2.putText(im,f"FPS : {int(1/(et-st))}",(400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Image",im)
        k = cv2.waitKey(10)
        if(k==27):
            break
        elif(k==32):
            cv2.waitKey(-1)