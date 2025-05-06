import mediapipe as mp
import cv2

CAMINHO = "./videos_fatiados/15/pessoa2video7-15.mp4"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)


cap = cv2.VideoCapture(CAMINHO)
acaba = False
letras_por_frame = []
count = 0

while not acaba:
    success, frame = cap.read()

    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_world_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("capture image", resized)
            count += 1
            print(f"\rFrame {count} : ", end="")
            classificacao = chr(cv2.waitKey(0))
            print(f"\rFrame {count} : {classificacao}")
            letras_por_frame.append(classificacao)
    else:
        print("Fim")
        acaba = True

