import cv2
import mediapipe as mp
import numpy as np

vetores_palavras = {
    "amanda" : [],
    "beatriz": [],
    "camila": [],
    "dublin": [],
    "eduardo": [],
    "fernando": [],
    "gabriela": [],
    "helsinque": [],
    "itália": [],
    "jacarta": [],
    "kyoto": [],
    "luxemburgo": [],
    "mariana": [],
    "nairobi": [],
    "oslo": [],
    "portugal": [],
    "qatar": [],
    "raphael": [],
    "samuel": [],
    "tokyo": [],
    "uruguai": [],
    "vancouver": [],
    "william": [],
    "xangai": [],
    "yuri": [],
    "zimbabwe": []
}

numeros_palavras = {
    1: "amanda",
    2: "beatriz",
    3: "camila",
    4: "dublin",
    5: "eduardo",
    6: "fernando",
    7: "gabriela",
    8: "helsinque",
    9 :"itália",
    10: "jacarta",
    11: "kyoto",
    12: "luxemburgo",
    13: "mariana",
    14: "nairobi",
    15: "oslo",
    16: "portugal",
    17: "qatar",
    18: "raphael",
    19: "samuel",
    20: "tokyo",
    21: "uruguai",
    22: "vancouver",
    23: "william",
    24: "xangai",
    25: "yuri",
    26: "zimbabwe",
}

videosFaltando = ["pessoa4video10-03", "pessoa1video7-08", "pessoa3video8-22"]

CAMINHO = './videos_fatiados/'

for palavra in range(1, 27):
    PASTA = f"{palavra}/"
    
    for pessoa in range(1, 5):
        PESSOA = f"pessoa{pessoa}"
        
        for video in range(1, 11):
            VIDEO = f"video{video}"
            
            if palavra < 10:
                PALAVRA = f"-0{palavra}.mp4"
            else:
                PALAVRA = f"-{palavra}.mp4"

            CAMINHO_TOTAL = CAMINHO + PASTA + PESSOA + VIDEO + PALAVRA

            if PESSOA + VIDEO + PALAVRA in videosFaltando: continue
            
            cap = cv2.VideoCapture(CAMINHO_TOTAL)

            mp_drawing = mp.solutions.drawing_utils

            mp_hands = mp.solutions.hands
            hand = mp_hands.Hands(max_num_hands=1)

            count_frames = 0
            acaba = False

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

                
                        vetores_palavras[numeros_palavras[palavra]].append(atual_point)      
                        count_frames += 1
                        print(f"\r{PESSOA+VIDEO+PALAVRA} -> Frames coletados: {count_frames}", end="")

                    cv2.imshow("capture image", frame)
                    cv2.waitKey(1)
                else:
                    acaba = True
                    print("\nVideo concluido!")
                    
            cv2.destroyAllWindows()


nome_arquivo = "dados.py"

with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
    # Escreve o dicionário como uma variável
    arquivo.write(f"vetores_palavras = {repr(vetores_palavras)}\n")