import cv2
import mediapipe as mp
import math

import letters_numbers as ln

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)

while True:
  success, frame = cap.read()

  if success:
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)
    if result.multi_hand_landmarks:
      min_distance = 1000
      letter = ""
      handLandmarks = result.multi_hand_landmarks[0]
      for chave, valor in ln.vetor.items():
        soma = 0
        for i in range(0, len(valor)):
          distance = math.sqrt((handLandmarks.landmark[i].x - valor[i][0]) ** 2 + (handLandmarks.landmark[i].y - valor[i][1]) ** 2 + (handLandmarks.landmark[i].z - valor[i][2]) ** 2)
          soma += distance
          
        soma = soma / 21
        if soma < min_distance:
          min_distance = soma
          letter = chave
          
      print(letter)
          
      #for hand_landmarks in result.multi_hand_landmarks:
        #print(hand_landmarks)
        #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
    cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord('q'):
      #print(hand_landmarks)
      break
    
cv2.destroyAllWindows