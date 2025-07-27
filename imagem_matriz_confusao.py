import numpy as np
import matplotlib.pyplot as plt
import dados.matriz_pedro_pedro as m

# Sua matriz (copie para cá ou carregue de um arquivo)
matriz = m.matriz

matriz_log = np.log1p(matriz)

plt.figure(figsize=(15, 12))
plt.imshow(matriz_log, interpolation='nearest', cmap='viridis')
plt.colorbar(label='log(1 + contagem)')

labels = [chr(i) for i in range(ord('a'), ord('z')+1)] + ['*']
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=90)
plt.yticks(tick_marks, labels)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão (escala logarítmica + valores)')

for i in range(matriz.shape[0]):
    for j in range(matriz.shape[1]):
        val = matriz[i, j]
        if val > 0:
            plt.text(j, i, str(val), ha='center', va='center', color='black', fontsize=6)

plt.savefig('dados/matriz_pedro_pedro.png', dpi=300, bbox_inches='tight')
plt.close()
