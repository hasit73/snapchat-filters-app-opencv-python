import cv2
import mediapipe as mp

class FaceAnalyzer:
    def __init__(self):
        self.__mp_draw = mp.solutions.drawing_utils
        self.__mp_face_mesh = mp.solutions.face_mesh

        self.__face_detector = self.__mp_face_mesh.FaceMesh(static_image_mode=False,min_detection_confidence = 0.6,min_tracking_confidence = 0.6)
        self.__face_mesh_marks = []
        self.face_points = None
        self.left_eye_points = [[144,160],
                    [145,159],
                    [153,158],
                    [154,157],
                    [161,163] ]

                    
        self.right_eye_points = [ [381,384],
                            [380,385],
                            [374,386],
                            [373,387],
                            [388,390] ]


    def get_face_points(self,image):
        self.face_points = []
        self.__face_mesh_marks = []
        results = self.__face_detector.process(image)
        ## Total 468 landmarks 
        if(results.multi_face_landmarks):
            for face in results.multi_face_landmarks:
                for lm in face.landmark:
                    x = lm.x
                    y = lm.y
                    h,w,c = image.shape
                    relative_x = int(x*w)
                    relative_y = int(y*h)
                    self.face_points.append([relative_x,relative_y])
                self.__face_mesh_marks.append(face)