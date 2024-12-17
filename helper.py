import dados as d

dados = d.dado

def funcao(hand_landmarks):
    landmarks = []
    for i in range(21):
        coordenadas = []
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        z = hand_landmarks.landmark[i].y
        coordenadas.append(x)
        coordenadas.append(y)
        coordenadas.append(z)

        landmarks.append(coordenadas)
    
    dados.append(landmarks)
    print("dados salvos")
    
def salvar_dados():
    with open("dados.py", "w") as f:
        f.write(f"dado = {d.dado}\n")