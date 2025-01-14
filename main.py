import cv2
import mediapipe as mp
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import letters_numbers as ln

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)

count = 0
lista = []

x = []  # Vetores de landmarks
y = []  # vetor de letras

for chave, valores in ln.vetor.items():
  for valor in valores:
    x.append(np.array(valor).flatten())
    y.append(chave)
    
x = np.array(x)
y = np.array(y)

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar o modelo k-NN
k = 5  # Número de vizinhos
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# Avaliar o modelo
y_pred = knn.predict(x_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")

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
        
      atual_point = np.array(atual_point)
          
      previsao = knn.predict([atual_point])
      print(f"Letra prevista: {previsao[0]}")
          
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