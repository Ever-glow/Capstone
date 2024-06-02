import numpy as np
import dlib
import cv2
import time
import winsound as sd

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness Detection'

face_cascade_name = "C:\prt\haarcascade_frontalface_default.xml" # PC 환경에 따라 수정
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = "C:\prt\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat" # PC 환경에 따라 수정
predictor = dlib.shape_predictor(predictor_file)

number_closed = 0
min_EAR = 0.22 # 눈 감음으로 인식할 기준, 인식 정도에 따라 수정
closed_limit = 22 # 졸음으로 인식할 눈 감음 count, 이 경우 2.2초이 인식 정도에 따라 수정
show_frame = None
sign = None
color = None

# 졸음 감지 시 알림음
def beepsound():
    fr = 1000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
# 눈 좌표 계산
def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)
# 졸음 감지 함수
def detectAndDisplay(image):
    global number_closed
    global color
    global show_frame
    global sign
    global detect

    image = cv2.resize(image, (frame_width, frame_height))
    show_frame = image
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        rect = dlib.rectangle(int(x), int(y), int(x + w),
         int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

        right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
        left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0,0], left_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for (i, point) in enumerate(show_parts):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            
        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 3
            if( number_closed<0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'sleep'
            number_closed = number_closed + 1
                     
        sign = 'sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

        # 졸음 감지시 화면을 gray 스케일 변환, 감지되었음을 알리는 detect 변수
        if( number_closed > closed_limit ):
            #show_frame = frame_gray
            detect = 1

            detect_text="drowsiness detected!"
            org=(50,100)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(show_frame,detect_text,org,font,1,(255,0,0),2)
            
        # 평상 시 원활한 졸음 감지를 위해 화면에 메세지 출력
        # 졸음에서 깬 뒤에도 졸음 감지 다시 동작을 위해 detect = 0 초기화
        else:
            detect = 0
            usual_text1="Keep your posture"
            usual_text2="to detect drowsiness"
            org1=(50,50)
            org2=(50,100)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(show_frame,usual_text1,org1,font,1,(254,1,15),2)
            cv2.putText(show_frame,usual_text2,org2,font,1,(254,1,15),2)
    
    cv2.putText(show_frame, sign , (10,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow(title_name, show_frame)
    
# 웹캠 불러오기    
cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened:
    print('Could not open video')
    exit(0)

detect = 0
# 실시간 영상을 위한 while문
while True:
    ret, frame = cap.read()
    if frame is None:
        print('Could not read frame')
        cap.release()
        break
    #졸음 감지 함수 동작, 졸음 감지 시 알림음 출력
    detectAndDisplay(frame)        
    if detect == 1:
        beepsound()
        detect = 0

    # q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
