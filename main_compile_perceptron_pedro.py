import numpy as np
import vetores_separados as vs
import tensorflow as tf
from sklearn.model_selection import train_test_split

x = []  # Vetores de landmarks
y = []  # vetor de letras
alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25
}

for chave, valores in vs.vetor_world_pedro.items():
  for valor in valores:
    x.append(np.array(valor).flatten())
    y.append(alfabeto.get(chave))

x = np.array(x)
y = np.array(y)

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

NUMERO_LETRAS = 26
Z = 40 ##
ATIVACAO = 'relu'
EPOCHS=120

model = tf.keras.Sequential([
    tf.keras.layers.Dense(Z, activation=ATIVACAO), 
    tf.keras.layers.Dense(NUMERO_LETRAS, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


#########################TREINAMENTO###########################

model.fit(x_train, y_train, epochs=EPOCHS)

###########################TESTES##############################
print("#" * 30)

# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Perda no teste: {loss}")
print(f"Acurácia no teste: {accuracy * 100:.2f}%")

# Fazer predições no conjunto de teste
predictions = model.predict(x_test)

# Obter a classe prevista para cada exemplo
predicted_classes = np.argmax(predictions, axis=1)

# Comparar com as classes reais
for i in range(10):  # Exibir os 10 primeiros exemplos como exemplo
    print(f"Exemplo {i + 1}:")
    print(f"Classe verdadeira: {y_test[i]} ({list(alfabeto.keys())[y_test[i]]})")
    print(f"Classe prevista: {predicted_classes[i]} ({list(alfabeto.keys())[predicted_classes[i]]})")
    print("-" * 30)


model.save("modelo_pedro.keras")


