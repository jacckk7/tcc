import numpy as np
import vetores_separados as vs
import letters_numbers as ln
import tensorflow as tf
import main_grafico_loss as mgl
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Desabilita logs interativos do Keras
tf.keras.config.disable_interactive_logging()

# Anotações para testes
LIMITE_MENOR_LOSS = 1000
DADO = "breno"  # "pedro", "breno" ou "BP"

########-NÃO MUDAR-##########
NUMERO_LETRAS = 26
ATIVACAO = 'relu'
#############################

# Preparando os dados
x = []  # Vetores de landmarks
y = []  # Vetor de letras
alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25
}

vetor = None
if DADO == "pedro":
    vetor = vs.vetor_world_pedro.items()
elif DADO == "breno":
    vetor = vs.vetor_world_breno.items()
elif DADO == "BP":
    vetor = ln.vetor_word.items()
else:
    print("ERRO - coloque o dado correto")
    exit()

if input(f"Utilizando dados {DADO}. Confirmar (s/n): ") != "s":
    exit()

for chave, valores in vetor:
    for valor in valores:
        x.append(np.array(valor).flatten())
        y.append(alfabeto.get(chave))

x = np.array(x)
y = np.array(y)

# Dividir em treino, validaçãoo e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5)

# Redimensionar os dados para o formato [samples, timesteps, features]
timesteps = 1
x_train = x_train.reshape((x_train.shape[0], timesteps, x_train.shape[1]))
x_validation = x_validation.reshape((x_validation.shape[0], timesteps, x_validation.shape[1]))
x_test = x_test.reshape((x_test.shape[0], timesteps, x_test.shape[1]))

#########################TREINAMENTO###########################
n_neuronios = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]

for i in n_neuronios:
    print("#" * 30)
    print(f"TREINO N {i} - {DADO}")

    # Modelo de rede Elman com SimpleRNN
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(i, activation=ATIVACAO, return_sequences=False, input_shape=(timesteps, x_train.shape[2])),
        tf.keras.layers.Dense(NUMERO_LETRAS, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    menor_loss = 100
    menor_loss_epoch = 0
    count_loss = 0
    count_epoch = 0
    n_loss = []

    while True:
        count_epoch += 1
        print(f"\rTreinando modelo {i}n: {count_epoch}e", end="")
        model.fit(x_train, y_train, epochs=1, verbose=1)
        loss, accuracy = model.evaluate(x_validation, y_validation, verbose=1)
        n_loss.append(loss)

        if loss < menor_loss:
            menor_loss = loss
            menor_loss_epoch = count_epoch
            count_loss = 0
            model.save(f"modelos_gerados/modelo_elman_{DADO}_{i}_n.keras")
        else:
            count_loss += 1

        if count_loss > LIMITE_MENOR_LOSS:
            break

    print(f"\nNeurônios: {i}")
    print(f"Epoch menor: {menor_loss_epoch}")
    print(f"Menor loss: {menor_loss}")
    print(f"Acurácia: {accuracy}%")

    print(f"TESTE N {i} - {DADO}")

    model = load_model(f"modelos_gerados/modelo_elman_{DADO}_{i}_n.keras")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Loss no teste: {loss}")
    print(f"Acurácia no teste: {accuracy * 100:.2f}%")

    mgl.salvar_vetor(f"graph_elman_{DADO}_{i}_n", n_loss)
    print("#" * 30)
