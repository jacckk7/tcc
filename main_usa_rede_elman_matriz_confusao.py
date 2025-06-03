import cv2
import dados.videos_to_letters_breno_v2 as vtb
import dados.videos_to_letters_pedro as vtp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import pandas as pd

tf.keras.config.disable_interactive_logging()

alfabeto_invertido = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g',
    7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n',
    14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u',
    21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: '*'
}

MODELO = "breno"
DADO = "pedro"

alfabeto = {v: k for k, v in alfabeto_invertido.items()}

model = load_model("modelos_gerados/modelo_elman_TS12_breno_fold_tr73461v95te082_100_n.keras")
print("modelo carregado")

if DADO == "pedro":
  videos_vetores = vtp.vetores
  videos_classificacoes = vtp.classificacao
elif DADO == "breno":
  videos_vetores = vtb.vetores
  videos_classificacoes = vtb.classificacao
else:
  print("ERRO - coloque o dado correto")
  exit()
if input(f"Utilizando modelo {MODELO} e dados {DADO}. Confirmar (s/n): ") != "s":
  exit()

# Acumuladores de rótulos para todos os vídeos
todos_rotulos_preditos = []
todos_rotulos_esperados = []

# Tamanho da janela
tamanho_janela = 25

idx_validos = [0, 2, 8]

# Loop sobre todos os vídeos disponíveis ou loop sobre os indices de teste
for idx in range(len(videos_vetores)):
#for idx in idx_validos:
  print(f"Fazendo teste com modelo do {MODELO} e dados do {DADO} para o index {idx}")
  entrada_frames = np.array(videos_vetores[idx])
  rotulos_esperados = np.array([alfabeto[letra] for letra in videos_classificacoes[idx]])

  # Padding dos vetores de entrada
  total_frames = len(entrada_frames)
  resto = total_frames % tamanho_janela
  if resto != 0:
    falta = tamanho_janela - resto
    padding_frames = np.zeros((falta, 63))
    entrada_frames = np.concatenate([entrada_frames, padding_frames], axis=0)

    padding_labels = np.full((falta,), -1)  # Rótulo inválido
    rotulos_esperados = np.concatenate([rotulos_esperados, padding_labels], axis=0)

  # Lista para guardar resultados deste vídeo
  rotulos_preditos = []

  for i in range(0, len(entrada_frames), tamanho_janela):
    entrada = entrada_frames[i:i+tamanho_janela].reshape(1, tamanho_janela, 63)
    previsao = model.predict(entrada, verbose=0)
    classes_previstas = np.argmax(previsao, axis=2)[0]
    rotulos_preditos.extend(classes_previstas)

  # Remove padding e acumula
  rotulos_preditos = np.array(rotulos_preditos)
  rotulos_esperados = np.array(rotulos_esperados)
  mask_validos = rotulos_esperados != -1

  todos_rotulos_preditos.extend(rotulos_preditos[mask_validos])
  todos_rotulos_esperados.extend(rotulos_esperados[mask_validos])

# Cria matriz de confusãoo
matriz_confusao = confusion_matrix(
  todos_rotulos_esperados,
  todos_rotulos_preditos,
  labels=list(range(27))
)

# Exibe a matriz
print("Matriz de Confusão (27x27):")
print(matriz_confusao)

# Exibe matriz com letras (opcional)
rotulos_letras = [alfabeto_invertido[i] for i in range(27)]
df = pd.DataFrame(matriz_confusao, index=rotulos_letras, columns=rotulos_letras)
print("\nMatriz de confusão com letras:")
print(df)

with open(f"./dados/matriz_{MODELO}_{DADO}.py", "w", encoding="utf-8") as arquivo:
  arquivo.write(f"matriz = {repr(matriz_confusao)}\n")
