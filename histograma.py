import dados.videos_to_letters_fluente as vl
import matplotlib.pyplot as plt
import numpy as np

#QUANTIDADE DE VIDEOS
QV = 17

alfabeto = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
    'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13,
    'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
    'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '*': 26
}

contagem_alfabeto = {
    'a': [0] * QV, 'b': [0] * QV, 'c': [0] * QV, 'd': [0] * QV, 'e': [0] * QV, 'f': [0] * QV, 'g': [0] * QV,
    'h': [0] * QV, 'i': [0] * QV, 'j': [0] * QV, 'k': [0] * QV, 'l': [0] * QV, 'm': [0] * QV, 'n': [0] * QV,
    'o': [0] * QV, 'p': [0] * QV, 'q': [0] * QV, 'r': [0] * QV, 's': [0] * QV, 't': [0] * QV, 'u': [0] * QV,
    'v': [0] * QV, 'w': [0] * QV, 'x': [0] * QV, 'y': [0] * QV, 'z': [0] * QV
}

vetor = [0] * 26
count_video = -1

for vetores in vl.classificacao:
    count_video += 1
    for letra in vetores:
        idx = alfabeto[letra]
        if idx < 26:
            contagem_alfabeto[letra][count_video] += 1
            vetor[idx] += 1

medias = []
for letra in list(alfabeto.keys())[:26]:
    valores = [v for v in contagem_alfabeto[letra] if v > 0]
    if valores:
        media = sum(valores) / len(valores)
    else:
        media = 0
    medias.append(media)

letras = list(alfabeto.keys())[:26]
letras_com_medias = list(zip(letras, medias))
letras_com_medias.sort(key=lambda x: x[1])

letras_ordenadas = [item[0] for item in letras_com_medias]
medias_ordenadas = [item[1] for item in letras_com_medias]
media_geral = sum(medias_ordenadas) / len(medias_ordenadas)

minimos_ordenados = []
for letra in letras_ordenadas:
    valores = [v for v in contagem_alfabeto[letra] if v > 0]
    if valores:
        minimos_ordenados.append(min(valores))
    else:
        minimos_ordenados.append(0)

maximos_ordenados = [max(contagem_alfabeto[letra]) for letra in letras_ordenadas]
erros_inferiores = [media - minimo for media, minimo in zip(medias_ordenadas, minimos_ordenados)]
erros_superiores = [maximo - media for media, maximo in zip(medias_ordenadas, maximos_ordenados)]

x_posicoes = np.arange(len(letras_ordenadas))
plt.figure(figsize=(12, 6))

plt.bar(x_posicoes, medias_ordenadas, color='skyblue', label='Média')
plt.errorbar(x_posicoes, medias_ordenadas, yerr=[erros_inferiores, erros_superiores],
             fmt='o', ecolor='black', elinewidth=1.5, capsize=5, label='Min-Max')

for i, letra in enumerate(letras_ordenadas):
    x = x_posicoes[i]
    media = medias_ordenadas[i]
    minimo = minimos_ordenados[i]
    maximo = maximos_ordenados[i]
    plt.text(x, minimo - 0.800, f'{minimo}', ha='center', va='top', fontsize=8, color='black')
    plt.text(x, maximo + 0.002, f'{maximo}', ha='center', va='bottom', fontsize=8, color='black')

plt.axhline(y=media_geral, color='red', linestyle='--', linewidth=1.5, label=f'Média geral = {media_geral:.2f}')
plt.xticks(x_posicoes, letras_ordenadas)
plt.xlabel("Letras")
plt.ylabel("Média de Ocorrências por Vídeo")
plt.title("Média, Mínimo e Máximo de Ocorrência de Cada Letra (ordenado)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Salvar o gráfico como imagem
plt.savefig('histograma_fluente.png', dpi=300)
plt.close()