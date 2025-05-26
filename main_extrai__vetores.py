import cv2
import mediapipe as mp
import numpy as np


NOME_ARQUIVO_VIDEO = 'IMG_0091.MOV'

cap = cv2.VideoCapture(NOME_ARQUIVO_VIDEO)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)


vetores_frames = []
count_frames = 0
acaba = False

while not acaba:
    success, frame = cap.read()

    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_world_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handLandmarks = result.multi_hand_world_landmarks[0]
      
            atual_point = []
      
            for i in range(0, 21):
                atual_point.append(handLandmarks.landmark[i].x)
                atual_point.append(handLandmarks.landmark[i].y)
                atual_point.append(handLandmarks.landmark[i].z)

      
            vetores_frames.append(atual_point)      
            count_frames += 1
            print(f"\rFrames coletados: {count_frames}", end="")

        cv2.imshow("capture image", frame)
        cv2.waitKey(1)
    else:
        acaba = True
        print("\nVideo concluido!")


nome_arquivo = NOME_ARQUIVO_VIDEO.split(".")[0] + ".py"

with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
    # Escreve o dicionário como uma variável
    arquivo.write(f"vetores_frames = {repr(vetores_frames)}\n")

cv2.destroyAllWindows