import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
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

DADOS = "pedro"
TIMESTEPS = 8
LLM = "geminiV2"

resultado_final = ""
tempos_extracao = []
tempos_classificacao = []
tempos_pos_processamento = []

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)
#model = load_model("modelos_gerados/modelo_elman_TS6_fluente_fold_tr00031011080712131606v011504te02140509_100_n.keras")
model = load_model("modelos_gerados/modelo_elman_TS8_pedro_fold_tr0905030201v0700te060804_100_n.keras")
#model = load_model("modelos_gerados/modelo_elman_TS12_breno_fold_tr73461v95te082_100_n.keras")
#model = load_model("modelos_gerados/modelo_elman_TS25_breno_fold_tr63917v50te284_100_n.keras")
print("modelo carregado")

#for palavra in palavras:
for i in range(1):
  palavra = "luxemburgo"
  cap = cv2.VideoCapture(f"videos_alfabeto/luxemburgo_pedro.mp4")
  if not cap.isOpened():
      print("Erro ao abrir o video.")
      exit()
  print("video capturado")

  sequencia_total = []
  sequence = []
  ultima_letra = ""
  while True:
    inicio = time.perf_counter()
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
      fim = time.perf_counter()
      tempos_extracao.append(fim - inicio)
      sequence.append(atual_point)

      if len(sequence) == TIMESTEPS:
        inicio = time.perf_counter()
        entrada = np.array(sequence).reshape(1, TIMESTEPS, 63)
        previsao = model.predict(entrada, verbose=0)
        classes_previstas = np.argmax(previsao, axis=2)[0]

        letras_previstas = []
        for n in classes_previstas : letras_previstas.append(alfabeto_invertido[n])
        fim = time.perf_counter()
        tempos_classificacao.append(fim - inicio)
        print("Sequencia prevista: ", ''.join(letras_previstas))
        sequencia_total = sequencia_total + letras_previstas
        sequence = []     
    #cv2.imshow("capture image", frame)
    if cv2.waitKey(1) == ord('q'):
      break      
  cv2.destroyAllWindows()
  inicio = time.perf_counter()
  resultado_final = resultado_final + f"Palavra: {palavra}\n" + pos_process(sequencia_total) + "\n"
  fim = time.perf_counter()
  tempos_pos_processamento.append(fim - inicio)
  for l in sequencia_total:print(l, end="")
  print()
  


with open(f"resultados_TS{TIMESTEPS}_{DADOS}.txt", "w", encoding="utf-8") as arquivo:
  arquivo.write(resultado_final)

media = sum(tempos_extracao) / len(tempos_extracao)
logs = print(f"\nTempo médio extracao: {media * 1000:.3f} ms")

media = sum(tempos_classificacao) / len(tempos_classificacao)
logs = print(f"\nTempo médio classificacao: {media * 1000:.3f} ms")

media = sum(tempos_pos_processamento) / len(tempos_pos_processamento)
logs = print(f"\nTempo médio pos processamento: {media * 1000:.3f} ms")