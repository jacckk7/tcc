import cv2
import mediapipe as mp
import numpy as np

from tensorflow.keras.models import load_model

alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)

model = load_model("meu_modelo.keras")
print("---------deuboom-----------")

while True:
  success, frame = cap.read()

  if success:
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)
    if result.multi_hand_landmarks:
      handLandmarks = result.multi_hand_landmarks[0]
      
      atual_point = []
      
      for i in range(0, 21):
        atual_point.append(handLandmarks.landmark[i].x)
        atual_point.append(handLandmarks.landmark[i].y)
        atual_point.append(handLandmarks.landmark[i].z)

      atual_point = np.array(atual_point).reshape(1, -1)


      previsao = model.predict(atual_point)
      classe_prevista = np.argmax(previsao, axis=1)[0]
      letra_prevista = list(alfabeto.keys())[list(alfabeto.values()).index(classe_prevista)]
      print(f"Letra prevista: {letra_prevista}")
          
      #for hand_landmarks in result.multi_hand_world_landmarks:
        #print(hand_landmarks)
        #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
    cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord('q'):
      #print(hand_landmarks)
      #print(lista)
      break
    """ elif cv2.waitKey(1) == ord('p'):
      count += 1
      print(count)
      lista.append(lv.landmark_vetor(hand_landmarks)) """
    
cv2.destroyAllWindows