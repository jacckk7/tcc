import numpy as np
import vetores_palavras as vs
import tensorflow as tf
import main_grafico_loss as mgl
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Desabilita logs interativos do Keras
tf.keras.config.disable_interactive_logging()

# Constantes
LIMITE_MENOR_LOSS = 1000
NUMERO_PALAVRAS = 5
ATIVACAO = 'relu'
kfolds = 10

# Carregar vetores
dados = vs.vetor_palavras.items()
alfabeto = {"abacaxi": 0, "bolacha": 1, "caderno": 2, "diamante": 3, "esquilo": 4}

# Preparando os dados
sequencias = []
labels = []

for chave, valores in dados:
    for valor in valores:
        sequencias.append(np.array(valor))
        labels.append(alfabeto[chave])

max_timesteps = 266
sequencias_pad = pad_sequences(sequencias, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')

x = np.array(sequencias_pad)
y = np.array(labels)

# Redimensiona para (samples, timesteps, features)
x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

# Loop por número de neurônios
n_neuronios = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]

for n in n_neuronios:
    print("#" * 30)
    print(f"TREINAMENTO COM {n} NEURÔNIOS")

    kf = KFold(n_splits=kfolds, shuffle=True)
    losses = []
    accuracies = []
    fold_num = 1

    for train_index, val_index in kf.split(x):
        print(f"Fold {fold_num}/{kfolds}")
        fold_num += 1

        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(n, activation=ATIVACAO, return_sequences=False, input_shape=(x.shape[1], x.shape[2])),
            tf.keras.layers.Dense(NUMERO_PALAVRAS, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        menor_loss = 100
        count_loss = 0
        count_epoch = 0
        n_loss = []

        while True:
            count_epoch += 1
            print(f"\rTreinando modelo com {n} neurônios, época {count_epoch}", end="")
            model.fit(x_train, y_train, epochs=1, verbose=0)
            loss, acc = model.evaluate(x_val, y_val, verbose=0)
            n_loss.append(loss)

            if loss < menor_loss:
                menor_loss = loss
                count_loss = 0
                model.save(f"modelos_gerados/modelo_elman_palavras_{n}_fold{fold_num - 1}.keras")
            else:
                count_loss += 1

            if count_loss > LIMITE_MENOR_LOSS:
                break

        losses.append(menor_loss)
        accuracies.append(acc)
        mgl.salvar_vetor(f"graph_elman_palavras_{n}_fold{fold_num - 1}", n_loss)

    print(f"\nRESULTADOS GERAIS PARA {n} NEURÔNIOS:")
    print(f"Loss médio: {np.mean(losses):.4f}")
    print(f"Accuracy médio: {np.mean(accuracies) * 100:.2f}%")
    print("#" * 30)
