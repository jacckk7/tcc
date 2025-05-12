import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.keras.config.disable_interactive_logging()

alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25
}

cap = cv2.VideoCapture("videos_fatiados/pessoa2video3-15.mp4")
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()
print("video capturado")

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)

model = load_model("modelos_gerados/modelo_elman_150_n.keras")
print("modelo carregado")

sequence = []
ultima_letra = ""
while True:
  success, frame = cap.read()

  if not success or frame is None:
    print("erro ao ler frame")
    break
  RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hand.process(RGB_frame)
  if result.multi_hand_world_landmarks:
    handLandmarks = result.multi_hand_world_landmarks[0]
    
    atual_point = []
    
    for i in range(0, 21):
      atual_point.append(handLandmarks.landmark[i].x)
      atual_point.append(handLandmarks.landmark[i].y)
      atual_point.append(handLandmarks.landmark[i].z)

    sequence.append(atual_point)

    if len(sequence) == 10:
      entrada = np.array(sequence).reshape(1, 10, 63)
      previsao = model.predict(entrada, verbose=0)
      classes_previstas = np.argmax(previsao, axis=2)[0]

      letras_previstas = [list(alfabeto.keys())[idx] for idx in classes_previstas]

      print("Sequencia prevista: ", ''.join(letras_previstas))

      sequence = []     
  cv2.imshow("capture image", frame)
  if cv2.waitKey(1) == ord('q'):
    break

    
cv2.destroyAllWindows

#      if (letra_prevista != ultima_letra):
#        print(f"\rLetra prevista: {letra_prevista}", end="")
#        ultima_letra = letra_prevista