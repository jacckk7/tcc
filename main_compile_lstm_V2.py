## ESSE SCRIPT GERA MODELOS SEM TIMESTEPS 


import numpy as np
import videos_to_letters
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Desabilita logs interativos do Keras
tf.keras.config.disable_interactive_logging()
gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    print("Está utilizando GPU : ", gpu)
else:
    print("Não está utilizando GPU :(")


LIMITE_MENOR_LOSS = 1000
########-NÃO MUDAR-##########
NUMERO_LETRAS = 27
ATIVACAO = 'relu'
#############################

# Preparando os dados
x = []  # Vetores de landmarks
y = []  # Vetor de letras
alfabeto = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
    "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13,
    "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20,
    "v": 21, "w": 22, "x": 23, "y": 24, "z": 25, "*": 26 
}

videos_vetores = videos_to_letters.vetores
videos_classificacoes = videos_to_letters.classificacao
# Dividir em treino, validaçãoo e teste
videos_vetores_train, videos_vetores_test, videos_classificacoes_train, videos_classificacoes_test = train_test_split(videos_vetores, videos_classificacoes, test_size=0.4)
videos_vetores_validation, videos_vetores_test, videos_classificacoes_validation, videos_classificacoes_test = train_test_split(videos_vetores_test, videos_classificacoes_test, test_size=0.5)

# Função para transformar os dados de vídeos 
def preparar_dados(videos, labels):
    x, y = [], []
    for video, rotulos in zip(videos, labels):
        vetores = np.array(video)  # shape: (frames, 63)
        rotulos_idx = [alfabeto[r] for r in rotulos]
        x.append(vetores)
        y.append(rotulos_idx)
    return pad_dados(x, y)

def pad_dados(x, y):
    x_pad = pad_sequences(x, padding='post', dtype='float32')
    y_pad = pad_sequences(y, padding='post', value=26)  # 26 = índice de '*'
    return x_pad, y_pad

# Processando os dados
tx_train, ty_train = preparar_dados(videos_vetores_train, videos_classificacoes_train)
tx_validation, ty_validation = preparar_dados(videos_vetores_validation, videos_classificacoes_validation)
tx_test, ty_test = preparar_dados(videos_vetores_test, videos_classificacoes_test)

# Verifica os shapes
print("Train:", tx_train.shape, ty_train.shape)
print("Validation:", tx_validation.shape, ty_validation.shape)
print("Test:", tx_test.shape, ty_test.shape)

#########################TREINAMENTO###########################
n_neuronios = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]

for i in n_neuronios:
    print("#" * 30)
    print(f"TREINO N {i}")

    # Modelo de rede LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(None, tx_train[0].shape[1])),
        tf.keras.layers.LSTM(i, activation=ATIVACAO, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUMERO_LETRAS, activation='softmax'))
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

    while True:
        count_epoch += 1
        print(f"\rTreinando modelo {i}n: {count_epoch}e", end="")

        history = model.fit(
            tx_train,
            ty_train,
            epochs=1,
            batch_size=32,
            validation_data=(tx_validation, ty_validation),
            shuffle=False,
            verbose=1)
        
        loss = history.history['val_loss'][0]
        accuracy = history.history['val_accuracy'][0]

        if loss < menor_loss:
            menor_loss = loss
            menor_loss_epoch = count_epoch
            count_loss = 0
            model.save(f"modelos_gerados/modelo_lstm_v2_{i}_n.keras")
        else:
            count_loss += 1

        if count_loss > LIMITE_MENOR_LOSS:
            break

    print(f"\nNeurônios: {i}")
    print(f"Epoch menor: {menor_loss_epoch}")
    print(f"Menor loss: {menor_loss}")
    print(f"Acurácia: {accuracy}%")

    print(f"TESTE N {i}")

    model = load_model(f"modelos_gerados/modelo_lstm_v2_{i}_n.keras")
    loss, accuracy = model.evaluate(tx_test, ty_test, verbose=1)
    print(f"Loss no teste: {loss}")
    print(f"Acurácia no teste: {accuracy * 100:.2f}%")

    print("#" * 30)
