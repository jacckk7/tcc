import numpy as np
import dados.videos_to_letters_pedro as videos_to_letters_pedro
import dados.videos_to_letters_ABC_espelhado as videos_to_letters_ABC
import main_grafico_loss as mgl
import tensorflow as tf
from tensorflow.keras.models import load_model

# Desabilita logs interativos do Keras
tf.keras.config.disable_interactive_logging()
gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    print("Está utilizando GPU : ", gpu)
else:
    print("Não está utilizando GPU :(")


TIMESTEPS = 10   # Definindo o tamanho da janela (timesteps)
LIMITE_MENOR_LOSS = 5000
DADO = "ABC"

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

# Função para transformar os dados de vídeos em sequências fixas para RNN
def preparar_dados(videos, labels):
    x, y = [], []
    for video, rotulos in zip(videos, labels):
        vetores = np.array(video)  # shape: (frames, 63)
        rotulos_idx = [alfabeto[r] for r in rotulos]

        # Padding se necessário
        resto = len(vetores) % TIMESTEPS
        if resto != 0:
            padding_frames = TIMESTEPS - resto
            vetores = np.vstack([vetores, np.zeros((padding_frames, vetores.shape[1]))])
            rotulos_idx += [26] * padding_frames  # usa '*' como padding

        # Dividir em blocos de timesteps
        for i in range(0, len(vetores), TIMESTEPS):
            x.append(vetores[i:i+TIMESTEPS])
            y.append(rotulos_idx[i:i+TIMESTEPS])

    return np.array(x), np.array(y)

def determinar_fold_vt(fold_treino):
    x=-1
    y=-1
    for i in range(4):
        if i in fold_treino:
            continue
        if x==-1:
            x = i
            continue
        y=i
        break
    return x, y


videos_vetores = None
videos_classificacoes = None
if DADO == "pedro":
    videos_vetores = videos_to_letters_pedro.vetores
    videos_classificacoes = videos_to_letters_pedro.classificacao
elif DADO == "breno":
    #coloca seus dados aqui, breno
    #videos_vetores = seu_dado.vetores
    #videos_classificacoes = seu_dado.classificacao
    exit()
elif DADO == "ABC":
    videos_vetores = videos_to_letters_ABC.vetores
    videos_classificacoes = videos_to_letters_ABC.classificacao
else:
    print("ERRO - coloque o dado correto")
    exit()
if input(f"Utilizando dados {DADO} com TIMESTEPS = {TIMESTEPS}. Confirmar (s/n): ") != "s":
    exit()

ordem_kfold = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

for fold_treino in ordem_kfold:
    fold_validacao, fold_teste = determinar_fold_vt(fold_treino)

    print(f"treino: {fold_treino}  validacao: {fold_validacao}  teste: {fold_teste}")
    videos_vetores_train = []
    videos_classificacoes_train = []
    videos_vetores_test = []
    videos_classificacoes_test = []
    videos_vetores_validation = []
    videos_classificacoes_validation = []

    # Dividir em treino, validação e teste
    videos_vetores_train.append(videos_vetores[fold_treino[0]])
    videos_vetores_train.append(videos_vetores[fold_treino[1]])
    videos_classificacoes_train.append(videos_classificacoes[fold_treino[0]])
    videos_classificacoes_train.append(videos_classificacoes[fold_treino[1]])
    videos_vetores_test.append(videos_vetores[fold_teste])
    videos_classificacoes_test.append(videos_classificacoes[fold_teste])
    videos_vetores_validation.append(videos_vetores[fold_validacao])
    videos_classificacoes_validation.append(videos_classificacoes[fold_validacao])

    # Processando os dados
    tx_train, ty_train = preparar_dados(videos_vetores_train, videos_classificacoes_train)
    tx_validation, ty_validation = preparar_dados(videos_vetores_validation, videos_classificacoes_validation)
    tx_test, ty_test = preparar_dados(videos_vetores_test, videos_classificacoes_test)

    # Verifica os shapes
    print("Train:", tx_train.shape, ty_train.shape)
    print("Validation:", tx_validation.shape, ty_validation.shape)
    print("Test:", tx_test.shape, ty_test.shape)

    #########################TREINAMENTO###########################
    n_neuronios = [100]

    n_loss = []
    for i in n_neuronios:
        print("#" * 30)
        print(f"TREINO N {i} - configuracao {fold_treino}")

        # Modelo de rede Elman com SimpleRNN
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(i, activation=ATIVACAO, return_sequences=True, input_shape=(TIMESTEPS, tx_train.shape[2])),
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
            print(f"\rTreinando modelo {i}n configuracao {fold_treino}: {count_epoch}e", end="")

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

            n_loss.append(loss)

            if loss < menor_loss:
                menor_loss = loss
                menor_loss_epoch = count_epoch
                count_loss = 0
                model.save(f"modelos_gerados/modelo_elman_TS{TIMESTEPS}_{DADO}_fold_{fold_treino[0]}{fold_treino[1]}_{i}_n.keras")
            else:
                count_loss += 1

            if count_loss > LIMITE_MENOR_LOSS:
                break

        print(f"\nNeurônios: {i}")
        print(f"Fold: {fold_treino}")
        print(f"Epoch menor: {menor_loss_epoch}")
        print(f"Menor loss: {menor_loss}")
        print(f"Acurácia: {accuracy}%")

        print(f"TESTE N {i} - configuracao {fold_treino}")

        model = load_model(f"modelos_gerados/modelo_elman_TS{TIMESTEPS}_{DADO}_fold_{fold_treino[0]}{fold_treino[1]}_{i}_n.keras")
        loss, accuracy = model.evaluate(tx_test, ty_test, verbose=1)
        print(f"Loss no teste: {loss}")
        print(f"Acurácia no teste: {accuracy * 100:.2f}%")

        mgl.salvar_vetor(f"graph_elman_TS{TIMESTEPS}_{DADO}_fold_{fold_treino[0]}{fold_treino[1]}_{i}_n", n_loss)

        print("#" * 30)
