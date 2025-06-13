import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pos_processamento import pos_process

tf.keras.config.disable_interactive_logging()

alfabeto_invertido = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g',
    7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n',
    14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u',
    21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: '*'
}

palavras = [
  "amanda",
  "beatriz",
  "camila",
  "dublin",
  "eduardo",
  "fernando",
  "gabriela",
  "helsinque",
  "italia",
  "jacarta",
  "kyoto",
  "luxemburgo",
  "mariana",
  "nairobi",
  "oslo",
  "portugal",
  "qatar",
  "raphael",
  "samuel",
  "tokyo",
  "uruguai",
  "vancouver",
  "william",
  "xangai",
  "yuri",
  "zimbabwe",
]

DADOS = "breno"
TIMESTEPS = 12
LLM = "gemini"

resultado_final = ""

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)
model = load_model("modelos_gerados/modelo_elman_TS12_breno_fold_tr73461v95te082_100_n.keras")
#model = load_model("modelos_gerados/modelo_elman_TS25_breno_fold_tr63917v50te284_100_n.keras")
print("modelo carregado")

for palavra in palavras:
  cap = cv2.VideoCapture(f"videos_palavras/{palavra}.MOV")
  if not cap.isOpened():
      print("Erro ao abrir o video.")
      exit()
  print("video capturado")

  sequencia_total = []
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

      if len(sequence) == TIMESTEPS:
        entrada = np.array(sequence).reshape(1, TIMESTEPS, 63)
        previsao = model.predict(entrada, verbose=0)
        classes_previstas = np.argmax(previsao, axis=2)[0]

        letras_previstas = []
        for n in classes_previstas : letras_previstas.append(alfabeto_invertido[n])

        #print("Sequencia prevista: ", ''.join(letras_previstas))
        sequencia_total = sequencia_total + letras_previstas
        sequence = []     
    resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("capture image", resized)
    if cv2.waitKey(1) == ord('q'):
      break

      
  cv2.destroyAllWindows()

  resultado_final = resultado_final + f"Palavra: {palavra}\n" + pos_process(sequencia_total)

with open(f"resultados_TS{TIMESTEPS}_{LLM}_{DADOS}.txt", "w", encoding="utf-8") as arquivo:
  arquivo.write(resultado_final)