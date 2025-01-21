import numpy as np
import vetores_separados as vs
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.keras.config.disable_interactive_logging()

#Anotações para testes
N_TESTE = 8
N_NEURONIOS = 128
DADO = "breno"
EPOCHS = 1000

########-NÃO MUDAR-##########
NUMERO_LETRAS = 26
ATIVACAO = 'relu'
#############################
x = []  # Vetores de landmarks
y = []  # vetor de letras
alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25
}

for chave, valores in vs.vetor_world_breno.items():
  for valor in valores:
    x.append(np.array(valor).flatten())
    y.append(alfabeto.get(chave))

x = np.array(x)
y = np.array(y)

# Dividir em treino e teste
# 40% Testes - 30% validacao - 30% Testes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(N_NEURONIOS, activation=ATIVACAO), 
    tf.keras.layers.Dense(NUMERO_LETRAS, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


#########################TREINAMENTO###########################
print("#" * 30)
print("TREINO")
menor_loss = 100
menor_i = -1


for i in range(EPOCHS//10):
    model.fit(x_train, y_train, epochs=10)
    loss, accuracy = model.evaluate(x_validation, y_validation, verbose=1)
    print(f"Epoch: {(i + 1) * 10}")
    print(f"loss: {loss}")
    print(f"Acurácia: {accuracy}%\n")
    if loss < menor_loss:
       menor_loss = loss
       menor_i = (i + 1)*10
       model.save(f"modelos_gerados/t{N_TESTE}_n{N_NEURONIOS}_e{menor_i}_{DADO}_modelo.keras")


print(f"menor loss: {menor_loss} ({menor_i})\n")
###########################TESTES##############################
print("#" * 30)
print("TESTE")
# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Loss: {loss}")
print(f"Acurácia no teste: {accuracy * 100:.2f}%")



