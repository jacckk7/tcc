import mediapipe as mp
import cv2
import dados.videos_to_letters_fluente as videos_to_letters

CAMINHO_VIDEO = "./videos_alfabeto/alfabeto-fluente-17.mp4"
CAMINHO_ARQUIVO = "./dados/videos_to_letters_fluente.py"

vetores = videos_to_letters.vetores
classificacoes = videos_to_letters.classificacao

print(f"Vídeo: {CAMINHO_VIDEO}")
print(f"Tamanho inicial vetores: {len(vetores)}")
print(f"Tamanho inicial classificacao: {len(classificacoes)}")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1)


cap = cv2.VideoCapture(CAMINHO_VIDEO)
acaba = False
classificacao_por_frame = []
vetores_por_frame = []
count = 0

while not acaba:
    success, frame = cap.read()

    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_world_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handLandmarks = result.multi_hand_world_landmarks[0]
                
            atual_point = []
    
            for i in range(0, 21):
                atual_point.append(handLandmarks.landmark[i].x)
                atual_point.append(handLandmarks.landmark[i].y)
                atual_point.append(handLandmarks.landmark[i].z)
            
            vetores_por_frame.append(atual_point)
            
            #resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("capture image", frame)
            count += 1
            print(f"\rFrame {count} : ", end="")
            classificacao = chr(cv2.waitKey(0))
            if classificacao == "!": exit()
            print(f"\rFrame {count} : {classificacao}")
            classificacao_por_frame.append(classificacao)
    else:
        print("Fim")
        acaba = True

vetores.append(vetores_por_frame)
classificacoes.append(classificacao_por_frame)

print("atualizando arquivo...")

with open(CAMINHO_ARQUIVO, "w", encoding="utf-8") as arquivo:
    arquivo.write(f"vetores = {repr(vetores)}\n")
    arquivo.write(f"classificacao = {repr(classificacoes)}\n")

print("Arquivo atualizado com sucesso!")
print(f"Tamanho final vetores: {len(vetores)}")
print(f"Tamanho final classificacao: {len(classificacoes)}")

